// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2011 Preferred Networks and Nippon Telegraph and Telephone Corporation.
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

#include <string>
#include "jubatus/util/data/string/utility.h"
#include "jubatus/util/lang/shared_ptr.h"
#include "jubatus/util/text/json.h"
#include "../common/exception.hpp"
#include "../common/jsonconfig.hpp"
#include "../nearest_neighbor/nearest_neighbor_factory.hpp"
#include "../unlearner/unlearner_factory.hpp"
#include "recommender_factory.hpp"
#include "recommender.hpp"

using std::string;
using jubatus::util::text::json::json;
using jubatus::util::lang::shared_ptr;
using jubatus::util::data::string::starts_with;
using jubatus::core::common::jsonconfig::config;
using jubatus::core::common::jsonconfig::config_cast_check;

namespace jubatus {
namespace core {
namespace recommender {
namespace {

const std::string NEAREST_NEIGHBOR_PREFIX("nearest_neighbor_recommender:");
struct nearest_neighbor_recommender_config {
  std::string method;
  config parameter;
  jubatus::util::data::optional<std::string> unlearner;
  jubatus::util::data::optional<config> unlearner_parameter;

  template<typename Ar>
  void serialize(Ar& ar) {
    ar & JUBA_MEMBER(method) & JUBA_MEMBER(parameter) &
        JUBA_MEMBER(unlearner) & JUBA_MEMBER(unlearner_parameter);
  }
};
}  // namespace

shared_ptr<recommender_base> recommender_factory::create_recommender(
    const string& name,
    const config& param,
    const string& id) {
  if (name == "nearest_neighbor_recommender") {
    nearest_neighbor_recommender_config conf =
        config_cast_check<nearest_neighbor_recommender_config>(param);
    if (conf.unlearner) {
      if (!conf.unlearner_parameter) {
        throw JUBATUS_EXCEPTION(
            common::config_exception() << common::exception::error_message(
                "unlearner is set but unlearner_parameter is not found"));
      }
      shared_ptr<unlearner::unlearner_base> unl(unlearner::create_unlearner(
          *conf.unlearner, common::jsonconfig::config(
              *conf.unlearner_parameter)));
      shared_ptr<nearest_neighbor::nearest_neighbor_base>
          nearest_neighbor_engine(nearest_neighbor::create_nearest_neighbor(
              conf.method, conf.parameter, id, unl));
      return shared_ptr<recommender_base>(
          new nearest_neighbor_recommender(nearest_neighbor_engine, unl));
    } else {
      shared_ptr<nearest_neighbor::nearest_neighbor_base>
          nearest_neighbor_engine(nearest_neighbor::create_nearest_neighbor(
              conf.method, conf.parameter, id));
      return shared_ptr<recommender_base>(
          new nearest_neighbor_recommender(nearest_neighbor_engine));
    }
  } else {
    throw JUBATUS_EXCEPTION(common::unsupported_method(name));
  }
}

}  // namespace recommender
}  // namespace core
}  // namespace jubatus

