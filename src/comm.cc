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

#include "comm.h"

#include <dlfcn.h>
#include <securec.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#include <stdexcept>
#include <sstream>
#include <string>
#include <mutex>
#include <vector>
#include <cstring>

#include "system.h"
#include "acl/acl_rt.h"

namespace dvm {
namespace {
const char *DEFAULT_SOCKET_IP = "127.0.0.1";
constexpr uint16_t DEFAULT_SOCKET_PORT = 10069;  // TODO: check whether mindspore use this port
const uint16_t MAX_BACK_LOG = 65535;

enum TopologyType : int64_t {
  TOPOLOGY_HCCS = 0,
  TOPOLOGY_PIX,
  TOPOLOGY_PIB,
  TOPOLOGY_PHB,
  TOPOLOGY_SYS,
  TOPOLOGY_SIO,
  TOPOLOGY_HCCS_SW
};

void DvmException(int rank_id, const char *error_str) {
  std::ostringstream oss;
  oss << "[" << rank_id << "]" << "DVM EXCEPTION. reason: " << error_str;
  throw std::runtime_error(oss.str());
}
}  // namespace

class SocketChannel {
 public:
  SocketChannel(int rank, int rank_size, int communicator_id);
  virtual ~SocketChannel();

  /**
   * @brief Gather data from send_buf to recv_buf
   * @param send_buf a pointer to send buffer
   * @param send_size size of send buffer
   * @param recv_buf a pointer to receive buffer
   */
  bool AllGather(const void *send_buf, size_t send_size, void *recv_buf);

 private:
  bool PreProcess();
  bool CreateServerSocket();
  bool CreateClientSocket();
  bool Listen();
  bool Accept();
  bool Connect();
  void CloseSocket();
  bool CheckErrno(int io_errno);
  int Send(int fd, const void *send_buf, size_t send_size, int flag);
  int Recv(int fd, void *recv_buf, size_t recv_size, int flag);
  bool ClientSendRecv(const uint8_t *send_buf, size_t send_size, uint8_t *recv_buf);
  bool ServerRecvSend(const uint8_t *send_buf, size_t send_size, uint8_t *recv_buf);
  bool IsServer();
  void GetIpAndPort();
  int AcceptConnection(int fd, sockaddr_in &client_addr, socklen_t *sin_size);

  int rank_;
  int rank_size_;
  int communicator_id_;
  int fd_;  // file descriptor created by socket
  std::vector<int> client_fds_;
  std::string ip_;
  uint16_t port_;
  bool is_init_ = false;
};

SocketChannel::SocketChannel(int rank, int rank_size, int communicator_id)
    : rank_(rank), rank_size_(rank_size), communicator_id_(communicator_id) {}

SocketChannel::~SocketChannel() { CloseSocket(); }

bool SocketChannel::CheckErrno(int io_errno) {
  return ((io_errno == EAGAIN) || (io_errno == EWOULDBLOCK) || (io_errno == EINTR));
}

bool SocketChannel::AllGather(const void *send_buf, size_t send_size, void *recv_buf) {
  auto usend_buf = reinterpret_cast<const uint8_t *>(send_buf);
  auto urecv_buf = reinterpret_cast<uint8_t *>(recv_buf);
  if (!is_init_ && !PreProcess()) {
    return false;
  }
  is_init_ = true;
  if (IsServer()) {
    return ServerRecvSend(usend_buf, send_size, urecv_buf);
  } else {
    return ClientSendRecv(usend_buf, send_size, urecv_buf);
  }
}

bool SocketChannel::PreProcess() {
  GetIpAndPort();
  if (IsServer()) {
    client_fds_.resize(rank_size_, -1);
    if (!CreateServerSocket()) {
      return false;
    }
    if (!Listen()) {
      return false;
    }
    if (!Accept()) {
      return false;
    }
  } else {
    if (!CreateClientSocket()) {
      return false;
    }
    if (!Connect()) {
      return false;
    }
  }
  return true;
}

bool SocketChannel::Listen() {
  if (listen(fd_, MAX_BACK_LOG) < 0) {
    return false;
  }

  return true;
}

bool SocketChannel::Accept() {
  struct sockaddr_in client_addr_;
  socklen_t sin_size = sizeof(struct sockaddr_in);

  for (int i = 1; i < rank_size_; ++i) {
    int fd = AcceptConnection(fd_, client_addr_, &sin_size);
    if (fd < 0) {
      return false;
    }

    int rank = 0;
    if (Recv(fd, &rank, sizeof(rank), 0) <= 0) {
      return false;
    }

    if (rank >= rank_size_ || rank <= 0 || client_fds_[rank] >= 0) {
      return false;
    }
    client_fds_[rank] = fd;
  }

  return true;
}

int SocketChannel::AcceptConnection(int fd, sockaddr_in &client_addr, socklen_t *sin_size) {
  int client_fd{0};
  struct sockaddr *client_addt_ptr = reinterpret_cast<struct sockaddr *>(&client_addr);

  do {
    client_fd = accept(fd, client_addt_ptr, sin_size);
    if (client_fd < 0) {
      if (!CheckErrno(errno)) {
        return -1;
      }
      continue;
    }
    break;
  } while (true);

  return client_fd;
}

bool SocketChannel::Connect() {
  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = inet_addr(ip_.c_str());
  addr.sin_port = htons(port_);

  int sleep_time = 1;
  int max_retry_cnt = 10;
  int retry_cnt = 0;
  bool success = false;
  struct sockaddr *addrPtr = reinterpret_cast<struct sockaddr *>(&addr);
  while (retry_cnt < max_retry_cnt) {
    if (connect(fd_, addrPtr, sizeof(struct sockaddr)) < 0) {
      if (errno == ECONNREFUSED) {
        retry_cnt++;
        sleep(sleep_time);
        continue;
      }
      if (errno != EINTR) {
        break;
      }
      continue;
    }
    success = true;
    break;
  }

  if (!success) {
    return false;
  }

  if (Send(fd_, &rank_, sizeof(rank_), 0) <= 0) {
    return false;
  }

  return true;
}

bool SocketChannel::CreateServerSocket() {
  fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (fd_ < 0) {
    return false;
  }

  int reuse = 1;
  if (setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(int)) < 0) {
    return false;
  }

  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = inet_addr(ip_.c_str());
  addr.sin_port = htons(port_);

  struct sockaddr *addr_ptr = reinterpret_cast<struct sockaddr *>(&addr);
  if (bind(fd_, addr_ptr, sizeof(struct sockaddr)) < 0) {
    return false;
  }
  return true;
}

bool SocketChannel::CreateClientSocket() {
  fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (fd_ < 0) {
    return false;
  }
  return true;
}

void SocketChannel::CloseSocket() {
  if (fd_ >= 0) {
    (void)close(fd_);
    fd_ = -1;
  }
  if (client_fds_.empty()) {
    return;
  }
  for (size_t i = 0; i < client_fds_.size(); i++) {
    if (client_fds_[i] >= 0) {
      (void)close(client_fds_[i]);
      client_fds_[i] = -1;
    }
  }
}

bool SocketChannel::IsServer() { return rank_ == 0; }

void SocketChannel::GetIpAndPort() {
  ip_ = DEFAULT_SOCKET_IP;
  port_ = DEFAULT_SOCKET_PORT + communicator_id_;
}

int SocketChannel::Send(int fd, const void *send_buf, size_t send_size, int flag) {
  do {
    auto ret = send(fd, send_buf, send_size, flag);
    if (ret < 0) {
      if (CheckErrno(errno)) {
        continue;
      }
    }
    return ret;
  } while (true);
}

int SocketChannel::Recv(int fd, void *recv_buf, size_t recv_size, int flag) {
  do {
    auto ret = recv(fd, recv_buf, recv_size, flag);
    if (ret < 0) {
      if (CheckErrno(errno)) {
        continue;
      }
    }
    return ret;
  } while (true);
}

bool SocketChannel::ClientSendRecv(const uint8_t *send_buf, size_t send_size, uint8_t *recv_buf) {
  if (Send(fd_, send_buf, send_size, 0) <= 0) {
    return false;
  }

  if (Recv(fd_, recv_buf, send_size * rank_size_, 0) <= 0) {
    return false;
  }

  return true;
}

bool SocketChannel::ServerRecvSend(const uint8_t *send_buf, size_t send_size, uint8_t *recv_buf) {
  memcpy_s(recv_buf, send_size, send_buf, send_size);

  for (int i = 1; i < rank_size_; ++i) {
    if (Recv(client_fds_[i], recv_buf + i * send_size, send_size, 0) <= 0) {
      return false;
    }
  }

  for (int i = 1; i < rank_size_; ++i) {
    if (Send(client_fds_[i], recv_buf, send_size * rank_size_, 0) <= 0) {
      return false;
    }
  }

  return true;
}

int Communicator::communicator_id_ = -1;

Communicator::Communicator(int rank_id, int rank_size) : inited_(false), rank_id_(rank_id), rank_size_(rank_size) {
  System::Instance().InitCommApi();
  communicator_id_++;
  socket_channel_ = new SocketChannel(rank_id_, rank_size_, communicator_id_);
  std::cout << "[" << rank_id_ << "] " << "load functions success" << std::endl;
}

Communicator::~Communicator() {
  FreeCommMem(peer_mem_[rank_id_]);
  delete socket_channel_;
}

void Communicator::InitMem() {
  auto ret = aclrtMalloc((void **)&peer_mem_[rank_id_], MAX_BUFFER_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    DvmException(rank_id_, "malloc shared mem failed");
  }

  ret = aclrtMemset(peer_mem_[rank_id_], MAX_BUFFER_BYTES, 0, MAX_BUFFER_BYTES);
  if (ret != ACL_SUCCESS) {
    DvmException(rank_id_, "memset shared mem falied");
  }
  return;
}

void Communicator::CollectDev() {
  (void)aclrtGetDevice(&dev_id_);
  // get other rank dev id, put into dev_list_
  bool ret = socket_channel_->AllGather(&dev_id_, sizeof(dev_id_), &dev_list_);
  if (!ret) {
    DvmException(rank_id_, "Collect device info failed");
  }
}

void Communicator::CollectPid(std::vector<uint32_t> &pids) {
  if (System::Instance().rtDeviceGetBareTGrid(&pids[rank_id_]) != ACL_SUCCESS) {
    DvmException(rank_id_, "DeviceGetBareTgid failed");
  }
  bool ret = socket_channel_->AllGather(&pids[rank_id_], sizeof(pids[rank_id_]), pids.data());
  if (!ret) {
    DvmException(rank_id_, "Collect pid failed");
  }
}

void Communicator::SetMemName(char *name) {
  if (System::Instance().rtIpcSetMemoryName(peer_mem_[rank_id_], MAX_BUFFER_BYTES, name, IPC_NAME_SIZE) != ACL_SUCCESS) {
    DvmException(rank_id_, "rtIpcSetMemoryName failed");
  }
}

void Communicator::SetIpcMemPid(const char *name, const std::vector<uint32_t> &pids) {
  for (int i = 0; i < rank_size_; i++) {
    if (rank_id_ == i) {
      continue;
    }
    int32_t pid_int32 = pids[i];
    if (System::Instance().rtSetIpcMemPid(name, &pid_int32, 1) != ACL_SUCCESS) {
      DvmException(rank_id_, "rtSetIpcMemPid failed");
    }
  }
}

void Communicator::CollectName(const char *name, char names[MAX_RANK_SIZE][IPC_NAME_SIZE]) {
  bool ret = socket_channel_->AllGather(name, IPC_NAME_SIZE, names);
  if (!ret) {
    DvmException(rank_id_, "Collect name failed");
  }
}

void Communicator::OpenIpcMem(const char names[MAX_RANK_SIZE][IPC_NAME_SIZE]) {
  static std::mutex mut;
  std::lock_guard<std::mutex> lock(mut);
  for (int i = 0; i < rank_size_; i++) {
    if (i == rank_id_) {
      continue;
    }
    int ret = System::Instance().rtIpcOpenMemory(reinterpret_cast<void **>(&peer_mem_[i]), names[i]);
    if (ret != ACL_SUCCESS) {
      std::stringstream oss;
      oss << "Open peer memory " << i << " failed, error code: " << ret;
      DvmException(rank_id_, oss.str().c_str());
    }
  }
}

void Communicator::InitCommon() {
  for (size_t i = 0; i < static_cast<size_t>(rank_size_); i++) {
    if (static_cast<size_t>(rank_id_) == i) {
      continue;
    }
    int64_t value = 0;
    if (System::Instance().rtGetPairDevicesInfo(rank_id_, i, 0, &value) != 0) {
      std::stringstream oss;
      oss << "no connection between " << rank_id_ << " and " << i;
      DvmException(rank_id_, oss.str().c_str());
    }
    // the connection type between 910B npus on a single machine must be fullmesh. value should always be 0 here.
    auto ret = aclrtDeviceEnablePeerAccess(i, 0);
    if (ret != ACL_SUCCESS) {
      DvmException(rank_id_, "aclrtDeviceEnablePeerAccess failed");
    }
  }
}

void Communicator::InitCommMem() {
  InitMem();

  std::vector<uint32_t> pids(MAX_RANK_SIZE);
  CollectPid(pids);

  char name[IPC_NAME_SIZE] = {};
  SetMemName(name);

  SetIpcMemPid(name, pids);
  char names[MAX_RANK_SIZE][IPC_NAME_SIZE] = {};
  CollectName(name, names);
  // For debug
  for (int i = 0; i < rank_size_; ++i) {
    std::cout << "[" << rank_id_ << "] " << "rank " << i << " mem name: " << names[i] << std::endl;
  }
  OpenIpcMem(names);
  // For debug
  for (int i = 0; i < rank_size_; ++i) {
    std::cout << "[" << rank_id_ << "] " << "peer_mem_ " << i << " addr: " << reinterpret_cast<void *>(peer_mem_[i])
              << std::endl;
  }
}

void Communicator::FreeCommMem(uint8_t *&mem) {
  if (mem != nullptr) {
    auto ret = aclrtFree(reinterpret_cast<void *>(mem));
    if (ret != ACL_SUCCESS) {
      DvmException(rank_id_, "aclrtFree failed");
    }
  }
  mem = nullptr;
}

bool Communicator::Init() {
  if (inited_) {
    return true;
  }
  CollectDev();
  InitCommon();
  InitCommMem();

  inited_ = true;
  return true;
}
}  // namespace dvm
