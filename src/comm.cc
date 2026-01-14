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
#include <cstring>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#include <stdexcept>
#include <sstream>
#include <string>
#include <mutex>
#include <vector>
#include <cstdlib>

#include "acl/acl_rt.h"
#include "system.h"

namespace dvm {
namespace {
const char *DEFAULT_SOCKET_IP = "127.0.0.1";
constexpr uint16_t DEFAULT_SOCKET_PORT = 10170;  // TODO: check whether mindspore use this port
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
  oss << "[" << rank_id << "]"
      << "DVM EXCEPTION. reason: " << error_str;
  throw std::runtime_error(oss.str());
}
}  // namespace

class SocketChannel {
 public:
  SocketChannel(int rank, int rank_size, const std::vector<uint32_t> &group_rank_list, int communicator_id);
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

  int rank_;  // local rank
  int rank_size_;
  int root_rank_;
  int communicator_id_;
  int fd_;  // file descriptor created by socket
  std::vector<int> client_fds_;

  std::string ip_;
  uint16_t port_;
  bool is_init_ = false;
};

SocketChannel::SocketChannel(int rank, int rank_size, const std::vector<uint32_t> &group_rank_list, int communicator_id)
    : rank_(rank), rank_size_(rank_size), communicator_id_(communicator_id) {
  root_rank_ = group_rank_list.size() == 0 ? 0 : group_rank_list[0];
}

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
    std::cerr << "[Server] listen() failed on fd " << fd_ << " with error: " << strerror(errno) << std::endl;
    return false;
  }

  return true;
}

bool SocketChannel::Accept() {
  struct sockaddr_in client_addr;
  socklen_t sin_size = sizeof(struct sockaddr_in);

  for (int i = 1; i < rank_size_; ++i) {
    int fd = AcceptConnection(fd_, client_addr, &sin_size);
    if (fd < 0) {
      std::cerr << "[Server] AcceptConnection failed at iteration " << i << ". errno: " << errno << " ("
                << strerror(errno) << ")" << std::endl;
      return false;
    }

    int rank = 0;
    int nrecv = Recv(fd, &rank, sizeof(rank), 0);
    if (nrecv <= 0) {
      std::cerr << "[Server] Failed to receive rank from fd=" << fd << ". recv returned " << nrecv
                << ", errno: " << errno << " (" << strerror(errno) << ")" << std::endl;
      return false;
    }

    if (rank >= rank_size_ || rank <= 0) {
      std::cerr << "[Server] Invalid rank " << rank << " (valid range: 1 ~ " << rank_size_ - 1 << ")" << std::endl;
      return false;
    }

    if (client_fds_[rank] >= 0) {
      std::cerr << "[Server] Duplicate connection attempt for rank=" << rank << ". Existing fd: " << client_fds_[rank]
                << ", new fd: " << fd << std::endl;
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
  struct sockaddr *addr_ptr = reinterpret_cast<struct sockaddr *>(&addr);

  while (retry_cnt < max_retry_cnt) {
    if (connect(fd_, addr_ptr, sizeof(struct sockaddr)) < 0) {
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
    std::cerr << "[Client] Failed to connect to " << ip_ << ":" << port_ << " after " << retry_cnt << " retries."
              << std::endl;
    return false;
  }

  int sent_bytes = Send(fd_, &rank_, sizeof(rank_), 0);
  if (sent_bytes <= 0) {
    std::cerr << "[Client] Failed to send rank=" << rank_ << " to server. Send returned " << sent_bytes
              << ", errno: " << errno << " (" << strerror(errno) << ")" << std::endl;
    return false;
  }

  return true;
}

bool SocketChannel::CreateServerSocket() {
  fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (fd_ < 0) {
    std::cerr << "[Server] Failed to create socket: " << strerror(errno) << std::endl;
    return false;
  }

  int reuse = 1;
  if (setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(int)) < 0) {
    std::cerr << "[Server] setsockopt(SO_REUSEADDR) failed: " << strerror(errno) << std::endl;
    return false;
  }

  if (setsockopt(fd_, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(int)) < 0) {
    std::cerr << "[Server] setsockopt(SO_REUSEPORT) failed: " << strerror(errno) << std::endl;
    return false;
  }

  struct sockaddr_in addr {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = inet_addr(ip_.c_str());
  addr.sin_port = htons(port_);

  struct sockaddr *addr_ptr = reinterpret_cast<struct sockaddr *>(&addr);
  if (bind(fd_, addr_ptr, sizeof(struct sockaddr)) < 0) {
    std::cerr << "[Client] bind() failed: " << strerror(errno) << " (ip=" << ip_ << ", port=" << port_ << ")"
              << std::endl;
    return false;
  }

  return true;
}

bool SocketChannel::CreateClientSocket() {
  fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (fd_ < 0) {
    std::cerr << "[Client] Failed to create socket: " << strerror(errno) << std::endl;
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
  port_ = DEFAULT_SOCKET_PORT + root_rank_;
}

int SocketChannel::Send(int fd, const void *send_buf, size_t send_size, int flag) {
  size_t total_sent = 0;
  const char *current_buf = static_cast<const char *>(send_buf);

  while (total_sent < send_size) {
    auto ret = send(fd, current_buf + total_sent, send_size - total_sent, flag);
    if (ret < 0) {
      int err = errno;
      std::cerr << "[Send] Failed to send to fd=" << fd << ", errno=" << err << " (" << strerror(err) << ")"
                << ", sent=" << total_sent << "/" << send_size << " bytes" << std::endl;
      return -1;
    } else if (ret == 0) {
      std::cerr << "[Send] send() returned 0 (fd=" << fd << "), connection may be closed by peer" << std::endl;
      break;
    }

    total_sent += ret;
  }

  return total_sent;
}

int SocketChannel::Recv(int fd, void *recv_buf, size_t recv_size, int flag) {
  size_t total_received = 0;
  char *current_buf = static_cast<char *>(recv_buf);

  while (total_received < recv_size) {
    auto ret = recv(fd, current_buf + total_received, recv_size - total_received, flag);
    if (ret < 0) {
      int err = errno;
      std::cerr << "[Recv] Failed to receive from fd=" << fd << ", errno=" << err << " (" << strerror(err) << ")"
                << ", received=" << total_received << "/" << recv_size << " bytes" << std::endl;
      return -1;
    } else if (ret == 0) {
      std::cerr << "[Recv] Connection closed by peer (fd=" << fd << "), total_received=" << total_received << "/"
                << recv_size << std::endl;
      break;
    }

    total_received += ret;
  }

  return total_received;
}

bool SocketChannel::ClientSendRecv(const uint8_t *send_buf, size_t send_size, uint8_t *recv_buf) {
  if (Send(fd_, send_buf, send_size, 0) <= 0) {
    std::cerr << "[Client] rank=" << rank_ << ", fd=" << fd_ << ": Failed to send data (" << send_size
              << " bytes), errno=" << errno << " (" << strerror(errno) << ")" << std::endl;
    return false;
  }

  if (Recv(fd_, recv_buf, send_size * rank_size_, 0) <= 0) {
    std::cerr << "[Client] rank=" << rank_ << ", fd=" << fd_ << ": Failed to receive data (" << send_size * rank_size_
              << " bytes), errno=" << errno << " (" << strerror(errno) << ")" << std::endl;
    return false;
  }

  return true;
}

bool SocketChannel::ServerRecvSend(const uint8_t *send_buf, size_t send_size, uint8_t *recv_buf) {
  std::memcpy(recv_buf, send_buf, send_size);

  for (int i = 1; i < rank_size_; ++i) {
    if (Recv(client_fds_[i], recv_buf + i * send_size, send_size, 0) <= 0) {
      std::cerr << "[Server] Failed to receive from rank=" << i << ", fd=" << client_fds_[i] << ", errno=" << errno
                << " (" << strerror(errno) << ")" << std::endl;
      return false;
    }
  }

  for (int i = 1; i < rank_size_; ++i) {
    if (Send(client_fds_[i], recv_buf, send_size * rank_size_, 0) <= 0) {
      std::cerr << "[Server] Failed to send to rank=" << i << ", fd=" << client_fds_[i] << ", errno=" << errno << " ("
                << strerror(errno) << ")" << std::endl;
      return false;
    }
  }
  return true;
}

int MemoryComm::communicator_id_ = -1;

MemoryComm::MemoryComm(int rank_id, int rank_size, const uint32_t *group_ranks) : inited_(false) {
  communicator_id_++;
  rank_id_ = rank_id;
  rank_size_ = rank_size;
  if (group_ranks == nullptr) {
    group_rank_list_ = {};
  } else {
    group_rank_list_ = std::vector<uint32_t>(group_ranks, group_ranks + rank_size);
  }
  peer_mem_ = peer_mem_data_;
}

MemoryComm::~MemoryComm() { FreeCommMem(); }

void MemoryComm::InitMem() {
  // step 1: reserve virtual memory address
  auto ret = aclrtReserveMemAddress((void **)&peer_mem_[rank_id_], MAX_BUFFER_BYTES, 0, nullptr, 1);
  if (ret != ACL_SUCCESS) {
    DvmException(rank_id_, "reserve virtual memory failed");
  }
  // step 2: malloc physical meomry
  aclrtPhysicalMemProp mem_property;
  mem_property.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  mem_property.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  mem_property.memAttr = ACL_HBM_MEM_HUGE;
  mem_property.location.id = dev_id_;
  mem_property.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  mem_property.reserve = 0;
  ret = aclrtMallocPhysical(&physical_mem_handle_, MAX_BUFFER_BYTES, &mem_property, 0);
  if (ret != ACL_SUCCESS) {
    DvmException(rank_id_, "malloc physical memory failed");
  }
  // step 3: map virtual memory addr to physic memory addr
  ret = aclrtMapMem(reinterpret_cast<void *>(peer_mem_[rank_id_]), MAX_BUFFER_BYTES, 0, physical_mem_handle_, 0);
  if (ret != ACL_SUCCESS) {
    DvmException(rank_id_, "map virtual memory addr to physical memory addr failed");
  }
  // step 4: clear memory. NOTICE: peer memory is ONLY cleared here
  ret = aclrtMemset(peer_mem_[rank_id_], MAX_BUFFER_BYTES, 0, MAX_BUFFER_BYTES);
  if (ret != ACL_SUCCESS) {
    DvmException(rank_id_, "memset shared mem falied");
  }
}

void MemoryComm::CollectDev() {
  int virtual_dev_id{0};
  (void)aclrtGetDevice(&virtual_dev_id);
  dev_id_ = virtual_dev_id;
  // get other rank dev id, put into dev_list_
  bool ret = socket_channel_->AllGather(&dev_id_, sizeof(dev_id_), &dev_list_);
  if (!ret) {
    DvmException(rank_id_, "Collect device info failed");
  }
}

void MemoryComm::CollectPid(std::vector<int32_t> &pids) {
  if (aclrtDeviceGetBareTgid(&pids[rank_id_]) != ACL_SUCCESS) {
    DvmException(rank_id_, "DeviceGetBareTgid failed");
  }
  bool ret = socket_channel_->AllGather(&pids[rank_id_], sizeof(pids[rank_id_]), pids.data());
  if (!ret) {
    DvmException(rank_id_, "Collect pid failed");
  }
}

void MemoryComm::CollectShareableHandle() {
  if (aclrtMemExportToShareableHandle(physical_mem_handle_, ACL_MEM_HANDLE_TYPE_NONE, 0, &peer_mem_handle_[rank_id_]) !=
      ACL_SUCCESS) {
    DvmException(rank_id_, "aclrtMemExportToShareableHandle failed");
  }
  bool ret =
    socket_channel_->AllGather(&peer_mem_handle_[rank_id_], sizeof(peer_mem_handle_[rank_id_]), peer_mem_handle_);
  if (!ret) {
    DvmException(rank_id_, "Collect name failed");
  }
}

void MemoryComm::SetPidToShareableHandle(std::vector<int32_t> &pids) {
  for (int i = 0; i < rank_size_; i++) {
    if (i == rank_id_) {
      continue;
    }
    // 01/26/2025: At present, each call of this interface can only add one pid to white list.
    if (aclrtMemSetPidToShareableHandle(peer_mem_handle_[rank_id_], &pids[i], 1) != ACL_SUCCESS) {
      DvmException(rank_id_, "aclrtMemSetPidToShareableHandle failed");
    }
  }
}

void MemoryComm::OpenIpcMem() {
  static std::mutex mut;
  std::lock_guard<std::mutex> lock(mut);
  for (int i = 0; i < rank_size_; i++) {
    if (i == rank_id_) {
      continue;
    }
    aclrtDrvMemHandle handle;
    if (auto ret = aclrtMemImportFromShareableHandle(peer_mem_handle_[i], dev_id_, &handle) != ACL_SUCCESS) {
      std::stringstream oss;
      oss << "aclrtMemImportFromShareableHandle failed, error code is: " << ret << ", device id is: " << dev_id_;
      DvmException(rank_id_, oss.str().c_str());
    }
    if (aclrtReserveMemAddress((void **)&peer_mem_[i], MAX_BUFFER_BYTES, 0, nullptr, 1) != ACL_SUCCESS) {
      DvmException(rank_id_, "reserve virtual memory failed");
    }
    if (aclrtMapMem(reinterpret_cast<void *>(peer_mem_[i]), MAX_BUFFER_BYTES, 0, handle, 0) != ACL_SUCCESS) {
      DvmException(rank_id_, "map virtual memory addr to physical memory addr failed");
    }
  }
}

void MemoryComm::InitCommon() {
  for (size_t i = 0; i < static_cast<size_t>(rank_size_); i++) {
    auto peer_dev_id = group_rank_list_.size() == 0 ? i : group_rank_list_[i];
    if (static_cast<size_t>(dev_id_) == peer_dev_id) {
      continue;
    }
    int32_t value = 0;
    aclrtDeviceCanAccessPeer(&value, dev_id_, peer_dev_id);
    if (value != 1) {
      std::stringstream oss;
      oss << "no connection between " << rank_id_ << " and " << i;
      DvmException(rank_id_, oss.str().c_str());
    }
    // the connection type between 910B npus on a single machine must be fullmesh. value should always be 0 here.
    auto ret = aclrtDeviceEnablePeerAccess(peer_dev_id, 0);
    if (ret != ACL_SUCCESS) {
      DvmException(rank_id_, "aclrtDeviceEnablePeerAccess failed");
    }
  }
}

void MemoryComm::InitCommMem() {
  InitMem();

  std::vector<int32_t> pids(MAX_RANK_SIZE);
  CollectPid(pids);

  CollectShareableHandle();

  SetPidToShareableHandle(pids);

  OpenIpcMem();
  // For debug
  for (int i = 0; i < rank_size_; ++i) {
    std::cout << "[" << rank_id_ << "] "
              << "peer_mem_ " << i << " addr: " << reinterpret_cast<void *>(peer_mem_[i]) << std::endl;
  }
}

void MemoryComm::FreeCommMem() {
  for (size_t i = 0; i < static_cast<size_t>(rank_size_); i++) {
    // step 1: unmap virtual memory address and physical memory space
    if (aclrtUnmapMem(reinterpret_cast<void *>(peer_mem_[i])) != ACL_SUCCESS) {
      DvmException(rank_id_, "aclrtUnmapMem failed");
    }
    // step 2: release virtual memory address
    if (aclrtReleaseMemAddress(reinterpret_cast<void *>(peer_mem_[i])) != ACL_SUCCESS) {
      DvmException(rank_id_, "aclrtReleaseMemAddress failed");
    }
  }
  // step 3: free physical memory. For each device, it only frees the physical memory allocated by itself
  if (aclrtFreePhysical(physical_mem_handle_) != ACL_SUCCESS) {
    DvmException(rank_id_, "aclrtFreePhysical failed");
  }
}

bool MemoryComm::Init() {
  if (inited_) {
    return true;
  }
  SocketChannel channel(rank_id_, rank_size_, group_rank_list_, communicator_id_);
  socket_channel_ = &channel;
  CollectDev();
  InitCommon();
  InitCommMem();
  socket_channel_ = nullptr;
  inited_ = true;
  return true;
}

HcclComm::HcclComm(void *hccl) : hccl_(hccl) {
  // TODO:
}

}  // namespace dvm
