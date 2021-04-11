#!/usr/bin/env bash
set -eu
sed "s%^//! kiddo = \".*\"%//! kiddo = \"$1\"%" < src/lib.rs > new-lib.rs
sed "s%docs\.rs/kiddo/.*\"%docs.rs/kiddo/$1\"%" < new-lib.rs > src/lib.rs
sed "s%^kiddo = \".*\"%kiddo = \"$1\"%" < README.md > new-README.md
mv new-README.md README.md
rm new-lib.rs