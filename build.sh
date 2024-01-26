#!/bin/bash
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

BASEPATH=$(cd "$(dirname $0)"; pwd)
OUTPUT_DIR="${BASEPATH}/output"

mk_new_dir()
{
  local create_dir="$1"

  if [[ -d "${create_dir}" ]]; then
    rm -rf "${create_dir}"
  fi

  mkdir -pv "${create_dir}"
}
mk_new_dir "${OUTPUT_DIR}"

write_checksum()
{
  PACKAGE_LIST=$(ls lib*.a) || exit
  for PACKAGE_NAME in $PACKAGE_LIST; do
    echo $PACKAGE_NAME
    sha256sum -b "$PACKAGE_NAME" >"$PACKAGE_NAME.sha256"
  done
}

TARGET_FILE="libdvm.a"
echo "---------------- build start ----------------"
source ${BASEPATH}/env.sh
rm -f *.o *.so *.a
make
if [ ! -f ${TARGET_FILE} ]; then
  echo "[ERROR] compile failed!"
  exit 1
fi
echo "---------------- build end ----------------"

# Write git info
git_branch=`git branch | awk '{print $2}'`
git_commit_id=`git rev-parse HEAD`
git_info_file="${OUTPUT_DIR}/lib_info.txt"
echo "[lib information]" > ${git_info_file}
echo "git branch: ${git_branch}" >> ${git_info_file}
echo "commit  id: ${git_commit_id}" >> ${git_info_file}

# Copy target to output directory
cp ${TARGET_FILE} ${OUTPUT_DIR}
cd ${OUTPUT_DIR}
write_checksum
