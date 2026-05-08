#pragma once

// Per-variant inline namespace for the C++ side of the CPU backend.
//
// In static multi-variant mode (GGML_CPU_ALL_VARIANTS_STATIC=ON), every
// variant compiles the same C++ TUs (traits.cpp, repack.cpp, ...) with
// different /arch flags. Their `extern "C"` symbols are renamed via the
// per-variant prefix header (cpu-mangling-<tag>.h), but C++-mangled symbols
// — out-of-line virtual destructors, explicit-instantiated templates,
// vtables — would still collide at link time.
//
// We resolve this by wrapping the contents of the `ggml::cpu` namespace
// (and the nested `ggml::cpu::repack` namespace) in a per-variant
// `inline namespace _v_<tag>`. Because the namespace is `inline`, name
// lookup is transparent within the variant's TU set: code that writes
// `ggml::cpu::tensor_traits` resolves identically. But the C++ mangled
// names contain the per-variant suffix, so distinct variants don't fight.
//
// In single-variant builds the macros are no-ops.

#if defined(GGML_CPU_STATIC_MULTI_VARIANT) && defined(GGML_CPU_VARIANT_TAG)
#  define GGML_CPU_VNS_CONCAT_(a, b) a##b
#  define GGML_CPU_VNS_CONCAT(a, b)  GGML_CPU_VNS_CONCAT_(a, b)
#  define GGML_CPU_VNS_NAME          GGML_CPU_VNS_CONCAT(_v_, GGML_CPU_VARIANT_TAG)
#  define GGML_CPU_VNS_BEGIN         inline namespace GGML_CPU_VNS_NAME {
#  define GGML_CPU_VNS_END           }
#else
#  define GGML_CPU_VNS_BEGIN
#  define GGML_CPU_VNS_END
#endif
