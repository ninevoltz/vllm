#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

events=()
for gpu in 1 2 3 4; do
    for link in 0 1 2 3 4 5; do
        events+=(-e "amdgpu_${gpu}/xgmi_link${link}_data_outbound/")
    done
done

sudo perf stat -a -I 1000 "${events[@]}"
