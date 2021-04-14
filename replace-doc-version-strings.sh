#!/usr/bin/env bash
set -eu
sed "s%^//! $1 = \".*\"%//! $1 = \"$2\"%" < src/lib.rs > new-lib.rs
sed "s%docs\.rs/$1/.*\"%docs.rs/$1/$2\"%" < new-lib.rs > src/lib.rs
sed "s%^$1 = \".*\"%$1 = \"$2\"%" < README.md > new-README.md
mv new-README.md README.md
rm new-lib.rs