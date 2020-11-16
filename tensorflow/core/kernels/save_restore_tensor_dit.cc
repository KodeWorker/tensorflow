/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/save_restore_tensor_dit.h"
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/util/tensor_slice_reader.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include "tensorflow/core/util/tensor_slice_writer.h"

#include "tensorflow/core/util/tensor_bundle/tensor_bundle_dit.h"

namespace tensorflow {

struct RestoreDITOp {
  RestoreDITOp& operator=(const RestoreDITOp&) = delete;

  bool should_run_in_pool(BundleReaderDIT* reader) const {
    TensorShape restored_full_shape;

    // Ignore status here; we'll catch the error later.
    if (!reader->LookupTensorShape(tensor_name, &restored_full_shape).ok()) {
      return false;
    }

    return restored_full_shape.num_elements() > kLargeShapeThreshold;
  }

  // Run this restore operation using a new BundleReader.
  void run_with_new_reader() {
    BundleReaderDIT reader(Env::Default(), reader_prefix);
    if (!reader.status().ok()) {
      status = reader.status();
      return;
    }

    status = run(&reader);
  }

  Status run(BundleReaderDIT* reader) {
    TensorShape restored_full_shape;
    TF_RETURN_IF_ERROR(
        reader->LookupTensorShape(tensor_name, &restored_full_shape));

    VLOG(1) << "Restoring tensor " << idx << " : " << tensor_name << " : "
            << restored_full_shape.num_elements();
    Tensor* restored_tensor;
    if (shape_and_slice.empty()) {
      // Lookup the full tensor.
      TF_RETURN_IF_ERROR(
          context->allocate_output(idx, restored_full_shape, &restored_tensor));
      TF_RETURN_IF_ERROR(reader->Lookup(tensor_name, restored_tensor));
    } else {
      // Lookup the slice.
      TensorShape parsed_full_shape;
      TensorSlice parsed_slice;
      TensorShape parsed_slice_shape;

      TF_RETURN_IF_ERROR(
          checkpoint::ParseShapeAndSlice(shape_and_slice, &parsed_full_shape,
                                         &parsed_slice, &parsed_slice_shape));

      if (!restored_full_shape.IsSameSize(parsed_full_shape)) {
        return errors::InvalidArgument(
            "tensor_name = ", tensor_name, "; shape in shape_and_slice spec ",
            parsed_full_shape.DebugString(),
            " does not match the shape stored in checkpoint: ",
            restored_full_shape.DebugString());
      }
      TF_RETURN_IF_ERROR(
          context->allocate_output(idx, parsed_slice_shape, &restored_tensor));
      TF_RETURN_IF_ERROR(
          reader->LookupSlice(tensor_name, parsed_slice, restored_tensor));
    }
    return Status::OK();
  }

  OpKernelContext* context;
  size_t idx;
  string tensor_name;
  string shape_and_slice;
  string reader_prefix;

  ::tensorflow::Status status;
};

}  // namespace

Status RestoreTensorsDIT(OpKernelContext* context, const Tensor& prefix,
                         const Tensor& tensor_names,
                         const Tensor& shape_and_slices,
                         gtl::ArraySlice<DataType> dtypes) {
  const string& prefix_string = prefix.scalar<tstring>()();

  const auto& tensor_names_flat = tensor_names.flat<tstring>();
  const auto& shape_and_slices_flat = shape_and_slices.flat<tstring>();

  // Sort lookup keys to improve locality when reading multiple tensors.
  std::vector<size_t> sorted_name_idx(tensor_names_flat.size());
  std::iota(sorted_name_idx.begin(), sorted_name_idx.end(), 0);
  std::sort(sorted_name_idx.begin(), sorted_name_idx.end(),
            [&tensor_names_flat](size_t a, size_t b) {
              return tensor_names_flat(a) < tensor_names_flat(b);
            });

  std::vector<std::unique_ptr<RestoreOp> > pool_restore_ops;
  std::vector<std::unique_ptr<RestoreOp> > direct_restore_ops;

  BundleReaderDIT default_reader(Env::Default(), prefix_string);
  TF_RETURN_IF_ERROR(default_reader.status());

  std::vector<string> mismatched_errors;
  for (const size_t i : sorted_name_idx) {
    TensorShape restored_full_shape;
    DataType original_dtype;
    const string& tensor_name = tensor_names_flat(i);
    TF_RETURN_IF_ERROR(default_reader.LookupDtypeAndShape(
        tensor_name, &original_dtype, &restored_full_shape));
    if (dtypes[i] != original_dtype) {
      string error_msg = strings::StrCat(
          "tensor_name = ", tensor_name, "; expected dtype ",
          DataTypeString(dtypes[i]), " does not equal original dtype ",
          DataTypeString(original_dtype));
      mismatched_errors.emplace_back(error_msg);
    }
  }
  if (!mismatched_errors.empty()) {
    const string error_msg = absl::StrJoin(mismatched_errors, "\n");
    return errors::InvalidArgument(error_msg);
  }

  for (auto i : sorted_name_idx) {
    const string& tensor_name = tensor_names_flat(i);
    const string& shape_and_slice = shape_and_slices_flat(i);
    auto op =
        new RestoreOp{context, i, tensor_name, shape_and_slice, prefix_string};
    if (op->should_run_in_pool(&default_reader)) {
      pool_restore_ops.emplace_back(op);
    } else {
      direct_restore_ops.emplace_back(op);
    }
  }

  {
    // Schedule any threaded operations first, skipping thread pool creation if
    // we don't have any expensive operations.
    std::unique_ptr<thread::ThreadPool> reader_pool;
    if (!pool_restore_ops.empty()) {
      reader_pool.reset(
          new thread::ThreadPool(Env::Default(), "restore_tensors", 8));
      for (auto& op : pool_restore_ops) {
        reader_pool->Schedule([&op]() { op->run_with_new_reader(); });
      }
    }

    // Read small tensors from the op thread
    for (auto& op : direct_restore_ops) {
      TF_RETURN_IF_ERROR(op->run(&default_reader));
    }
  }

  // Check status of pool ops; this must come after the pool shuts down.
  for (auto& op : pool_restore_ops) {
    TF_RETURN_IF_ERROR(op->status);
  }

  for (auto i : sorted_name_idx) {
    const string& tensor_name = tensor_names_flat(i);
    if (dtypes[i] != context->mutable_output(i)->dtype()) {
      return errors::InvalidArgument(
          "tensor_name = ", tensor_name, "; expected dtype ",
          DataTypeString(dtypes[i]), " does not equal restored dtype ",
          DataTypeString(context->mutable_output(i)->dtype()));
    }
  }

  return Status::OK();
}

}  // namespace tensorflow
