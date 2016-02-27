// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2014 Preferred Networks and Nippon Telegraph and Telephone Corporation.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License version 2.1 as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

#ifndef JUBATUS_CORE_FRAMEWORK_PACKER_HPP_
#define JUBATUS_CORE_FRAMEWORK_PACKER_HPP_

#include <msgpack.hpp>

namespace jubatus {
namespace core {
namespace framework {

class jubatus_writer {
 public:
  virtual ~jubatus_writer() {}
  virtual void write(const char* buf, unsigned int len) = 0;
};

class jubatus_packer {
 public:
  explicit jubatus_packer(jubatus_writer& w) : writer_(w) {
  }
  void write(const char* buf, std::size_t len) {
    writer_.write(buf, static_cast<unsigned int>(len));
  }
 private:
  jubatus_writer& writer_;
};

typedef msgpack::MSGPACK_DEFAULT_API_NS::packer<jubatus_packer> packer;

}  // namespace framework
}  // namespace core
}  // namespace jubatus

#endif  // JUBATUS_CORE_FRAMEWORK_PACKER_HPP_
