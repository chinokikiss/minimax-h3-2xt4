param([Parameter(Mandatory=$true)][string]$ExperimentRoot)
$ErrorActionPreference='Stop'
# Benchmark-only library build. No H3 sources/packages are replaced.
$root=(Resolve-Path -LiteralPath $ExperimentRoot).Path
$fse=Join-Path $root 'sources\zipnn-0.5.4\include\FiniteStateEntropy\lib'
$zig=Join-Path $root 'sources\openzl-native\src\openzl\codecs\zigzag'
$revision=git -C (Join-Path $root 'sources\openzl-native') rev-parse HEAD
if ($revision -ne '32246b48faee46807f84183dac4db479089f5445') {
    throw 'Unexpected OpenZL source revision'
}
$vswhere='C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$studio=& $vswhere -latest -property installationPath
& (Join-Path $studio 'Common7\Tools\Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$target=Join-Path $root 'native'
New-Item -ItemType Directory -Force $target | Out-Null
Push-Location -LiteralPath $target
try {
    # Unmodified upstream files. Export switches only expose their C ABI.
    & cl.exe /nologo /O2 /arch:AVX2 /LD /DFSE_DLL_EXPORT=1 `
        "$fse\fse_compress.c" "$fse\fse_decompress.c" `
        "$fse\entropy_common.c" "$fse\hist.c" "$fse\debug.c" `
        "$zig\encode_zigzag_kernel.c" "$zig\decode_zigzag_kernel.c" `
        /Fe:zigzag_fse.dll /link /EXPORT:ZL_zigzagEncode8 /EXPORT:ZL_zigzagDecode8
    if ($LASTEXITCODE) { throw 'Upstream codec library build failed' }
} finally { Pop-Location }
