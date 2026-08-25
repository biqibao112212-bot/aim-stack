#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 ABSOLUTE_DESTINATION/onnxruntime-linux-x64-1.22.1" >&2
  exit 2
fi

destination="$1"
expected_name="onnxruntime-linux-x64-1.22.1"
if [[ "$destination" != /* || "$(basename -- "$destination")" != "$expected_name" ]]; then
  echo "destination must be an absolute path ending in $expected_name" >&2
  exit 2
fi
if [[ -e "$destination" ]]; then
  echo "refusing to overwrite existing dependency: $destination" >&2
  exit 1
fi

for command_name in curl sha256sum unzip; do
  command -v "$command_name" >/dev/null || {
    echo "missing command: $command_name" >&2
    exit 1
  }
done

download_url="https://github.com/microsoft/onnxruntime/releases/download/v1.22.1/Microsoft.ML.OnnxRuntime.1.22.1.nupkg"
package_sha256="2ee0ed327f6cf2b860182bc4f2feb905c44a596cd120a05c510da6e4044a3e58"
task_temp_dir="$(mktemp -d)"
trap 'rm -r -- "$task_temp_dir"' EXIT
package_path="$task_temp_dir/Microsoft.ML.OnnxRuntime.1.22.1.nupkg"

curl --fail --location --retry 3 --output "$package_path" "$download_url"
printf '%s  %s\n' "$package_sha256" "$package_path" | sha256sum --check --status

install -d "$destination/include" "$destination/lib"
unzip -q -j "$package_path" 'build/native/include/*' -d "$destination/include"
unzip -q -j "$package_path" 'runtimes/linux-x64/native/*' -d "$destination/lib"
unzip -q -j "$package_path" LICENSE -d "$destination"
ln -s libonnxruntime.so "$destination/lib/libonnxruntime.so.1"

actual_library_sha256="$(sha256sum "$destination/lib/libonnxruntime.so" | awk '{print $1}')"
expected_library_sha256="3907398e408dae083deb3439e8f643d9e26180ed614b29cc7d5ec342ce5ce06f"
if [[ "$actual_library_sha256" != "$expected_library_sha256" ]]; then
  echo "installed libonnxruntime hash mismatch" >&2
  exit 1
fi

echo "installed ONNX Runtime 1.22.1 at $destination"
