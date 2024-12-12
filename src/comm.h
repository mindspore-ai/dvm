/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef _DVM_COMMUNICATOR_H_
#define _DVM_COMMUNICATOR_H_

#include <cstdint>
#include <vector>

#include "isa.h"

namespace dvm {
constexpr size_t MAX_RANK_SIZE = 8;
constexpr size_t MAX_BUFFER_BYTES = 204 * 1024 * 1024;  // 204MB
constexpr size_t IPC_NAME_SIZE = 65;
class SocketChannel;

class Communicator {
 public:
  Communicator(int rank_id, int rank_size);
  ~Communicator();
  bool Init();

  const std::vector<uint8_t *> GetPeerMemPtrs() const {
    return std::vector<uint8_t *>(peer_mem_, peer_mem_ + MAX_RANK_SIZE);
  }
  uint8_t *GetPeerMemPtr(size_t i) const { return peer_mem_[i % rank_size_]; };
  const int GetRankId() const { return rank_id_; }
  const int GetRankSize() const { return rank_size_; }

 private:
  /**
   * @brief Collect all the device ids
   */
  void CollectDev();

  /**
   * @brief Collect all the process ids
   * @param pids a reference to the vector which stores pid. After calling this method, this variable in each process
   * will get all pids of the current task
   */
  void CollectPid(std::vector<uint32_t> &pids);
  /**
   * @brief Collect all the peer memory names
   * @param name the name of peer memory which is allocated by the current process
   * @param names a buffer of names. After calling this method, this variable in each process will get all peer memory
   * names of the current task
   */
  void CollectName(const char *name, char names[MAX_RANK_SIZE][IPC_NAME_SIZE]);
  /**
   * @brief Set name of peer memory which is allocated by the current process
   * @param name a buffer of name. After calling this method, this buffer will be filled by the name returned by runtime
   * interface
   */
  void SetMemName(char *name);
  /**
   * @brief For each peer memory, add all pids of the current task to a white list
   * @param name the name of peer memory which is allocated by the current process
   * @param pids all pids of the current task
   */
  void SetIpcMemPid(const char *name, const std::vector<uint32_t> &pids);
  /**
   * @brief Enable each process to visit all the peer memories allocated by other process
   * @param names names of all the peer memories
   */
  void OpenIpcMem(const char names[MAX_RANK_SIZE][IPC_NAME_SIZE]);
  /**
   * @brief Malloc peer memory and clear it. Each process only process the peer memory of npu which it occupies
   */
  void InitMem();
  /**
   * @brief Initialize the connection between any two npus
   */
  void InitCommon();
  /**
   * @brief Initialize peer memory
   */
  void InitCommMem();
  /**
   * @brief Free peer memory. Each process only process the peer memory of npu which it occupies
   * @param mem the memory needs to be freed
   */
  void FreeCommMem(uint8_t *&mem);

  bool inited_;
  int rank_id_;    // global rank id
  int rank_size_;  // global rank size
  static int communicator_id_;
  int dev_id_;  // local device id, if all the npus are on the same
  uint8_t *peer_mem_[MAX_RANK_SIZE] = {};
  SocketChannel *socket_channel_;
  std::vector<int> dev_list_ = {};
};
}  // namespace dvm

#endif  // _DVM_COMMUNICATOR_H_
