# Description:
#   Tensor bundle: a module to efficiently serialize and deserialize tensors.

load(
    "//tensorflow:tensorflow.bzl",
    "cc_header_only_library",
    "if_not_windows",
    "tf_cc_test",
    "tf_copts",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# To be exported to tensorflow/core:mobile_srcs.
filegroup(
    name = "mobile_srcs",
    srcs = [
        "byte_swap.cc",
        "byte_swap.h",
        "naming.cc",
        "naming.h",
        "tensor_bundle.cc",
        "tensor_bundle.h",
		"tensor_bundle_dit.cc",
        "tensor_bundle_dit.h",
    ],
)

alias(
    name = "android_srcs",
    actual = ":mobile_srcs",
)

cc_library(
    name = "tensor_bundle",
    srcs = [
        "byte_swap.cc",
        "tensor_bundle.cc",
		"tensor_bundle_dit.cc",
    ],
    hdrs = [
        "byte_swap.h",
        "tensor_bundle.h",
		"tensor_bundle_dit.h",
    ],
    copts = tf_copts() + if_not_windows(["-Wno-sign-compare"]),
    deps = [
        ":naming",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_header_only_library(
    name = "tensor_bundle_headers_lib",
    features = ["-parse_headers"],  # Transitively pulls in Eigen headers
    deps = [":tensor_bundle"],
)

cc_library(
    name = "naming",
    srcs = ["naming.cc"],
    hdrs = ["naming.h"],
    deps = ["//tensorflow/core:lib"],
)

tf_cc_test(
    name = "tensor_bundle_test",
    srcs = ["tensor_bundle_test.cc"],
    data = glob(["testdata/**"]),
    tags = [
        "nomsan",
        "notsan",
    ],
    deps = [
        ":tensor_bundle",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
