// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2011,2012 Preferred Networks and Nippon Telegraph and Telephone Corporation.
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

#ifndef JUBATUS_CORE_COMMON_UNORDERED_MAP_HPP_
#define JUBATUS_CORE_COMMON_UNORDERED_MAP_HPP_

#include <msgpack.hpp>
#include "jubatus/util/data/unordered_map.h"

// to make util::data::unordered_map serializable

namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
namespace adaptor {

template <typename K, typename V, typename H, typename E, typename A>
struct convert<jubatus::util::data::unordered_map<K, V, H, E, A> > {
  msgpack::object const& operator()(
      msgpack::object const& o,
      jubatus::util::data::unordered_map<K, V, H, E, A>& v) const {
    if (o.type != type::MAP) {
      throw type_error();
    }
    object_kv* const p_end = o.via.map.ptr + o.via.map.size;
    for (object_kv* p = o.via.map.ptr; p != p_end; ++p) {
      K key;
      p->key.convert(&key);
      p->val.convert(&v[key]);
    }
    return o;
  }
};

template <typename K, typename V, typename H, typename E, typename A>
struct pack<jubatus::util::data::unordered_map<K, V, H, E, A> > {
  template <typename Stream>
  msgpack::packer<Stream>& operator()(
      msgpack::packer<Stream>& o,
      jubatus::util::data::unordered_map<K, V, H, E, A> const& v) const {
    o.pack_map(v.size());
    typedef typename
      jubatus::util::data::unordered_map<K, V, H, E, A>::const_iterator
      iter_t;
    for (iter_t it = v.begin(); it != v.end(); ++it) {
      o.pack(it->first);
      o.pack(it->second);
    }
    return o;
  }
};

}  // namespace adaptor
}  // MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
}  // namespace msgpack

#endif  // JUBATUS_CORE_COMMON_UNORDERED_MAP_HPP_
