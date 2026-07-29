/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _DVM_TILE_BUILDER_H_
#define _DVM_TILE_BUILDER_H_

#include "dvm.h"

namespace dvm {
class TileRef {
 public:
  enum { MAX_TILE_DIM = 4 };
  TileRef() = default;
  ~TileRef() = default;

  size_t dim_size{0};
  int64_t dims[MAX_TILE_DIM];

  void SetTail(int64_t tail) { tail_ = tail; }
  int64_t GetTail() const { return tail_; }

 protected:
  int64_t tail_{0};
};

typedef void* GmRef;

class TObject;
class TBuilder;

class TileBuilder {
 public:
  TileBuilder();
  ~TileBuilder();

  TObject *Load(DataType type, GmRef *gm, IntArrayRef *tile_shape, TileRef *tile_space, IntArrayRef *stride = nullptr);
  TObject *Store(TObject *input, GmRef *gm, TileRef *tile_space, IntArrayRef *stride = nullptr);

  template <UnaryType op_type>
  TObject *Unary(TObject *x);
  template <BinaryType op_type>
  TObject *Binary(TObject *lhs, TObject *rhs);

  void CodeGen(int64_t tile_space_size, int64_t block_dim = 0);
  int Launch(bool reloc, void *stream);

  int64_t MaxTileSize();

  const char *Dump() const;
  const char *Das() const;

 protected:
  TBuilder *impl_;
};
}  // namespace dvm
#endif  // _DVM_TILE_BUILDER_H_
