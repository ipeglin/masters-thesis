// Build script:
//
// 1. Embeds an `rpath` pointing at `$LIBTORCH/lib` so the linked binary can
//    locate `libtorch_*.{dylib,so}` at runtime without `DYLD_LIBRARY_PATH` /
//    `LD_LIBRARY_PATH` juggling. Works on both macOS (Mach-O LC_RPATH) and
//    Linux (DT_RUNPATH); the path is whatever `LIBTORCH` resolved to on the
//    machine where the binary was built, so each host (laptop, CentOS
//    cluster) bakes its own correct path. Skipped when `LIBTORCH` is unset
//    (e.g. classify-only builds with `--no-default-features`).
//
// 2. Windows-only `git2` link fix.
fn main() {
    if let Ok(libtorch) = std::env::var("LIBTORCH") {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target_os == "macos" || target_os == "linux" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{libtorch}/lib");
        }
        // Re-run if LIBTORCH moves or the libs change.
        println!("cargo:rerun-if-env-changed=LIBTORCH");
    }

    #[cfg(target_os = "windows")]
    println!("cargo:rustc-link-lib=advapi32");
}
