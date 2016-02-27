#include <jubatus/core/common/jsonconfig.hpp>
#include <jubatus/core/fv_converter/converter_config.hpp>
#include <jubatus/core/fv_converter/datum.hpp>
#include <jubatus/core/driver/anomaly.hpp>
#include <jubatus/core/driver/classifier.hpp>
#include <jubatus/core/driver/nearest_neighbor.hpp>
#include <jubatus/core/driver/recommender.hpp>
#include <jubatus/core/driver/regression.hpp>
#include <jubatus/core/driver/clustering.hpp>
#include <jubatus/core/driver/burst.hpp>
#include <jubatus/core/driver/bandit.hpp>
#include <jubatus/util/text/json.h>
#include <jubatus/core/storage/storage_factory.hpp>
#include <jubatus/core/classifier/classifier_factory.hpp>

using jubatus::util::lang::shared_ptr;
using jubatus::core::fv_converter::datum;
namespace jubacore = jubatus::core;
namespace jubacomm = jubatus::core::common;
namespace jubadriver = jubatus::core::driver;
namespace jubafvconv = jubatus::core::fv_converter;
namespace jubalang = jubatus::util::lang;
namespace jubajson = jubatus::util::text::json;
namespace jubaframework = jubatus::core::framework;

struct Handle {
    shared_ptr<jubadriver::classifier> handle;
    shared_ptr<std::string> config;
};

extern "C" {

Handle* create_classifier(const char *config_json_text) {
    auto *handle = new Handle;
    handle->config.reset(new std::string(config_json_text));
    jubajson::json cfg = jubalang::lexical_cast<jubajson::json>(*(handle->config));

    std::string method;
    jubafvconv::converter_config fvconv_config;
    jubacomm::jsonconfig::config params;

    method.assign(static_cast<jubajson::json_string*>(cfg["method"].get())->get());
    jubajson::from_json(cfg["converter"], fvconv_config);
    params = jubacomm::jsonconfig::config(cfg["parameter"]);

    handle->handle.reset(new jubadriver::classifier(
        jubacore::classifier::classifier_factory::create_classifier(
            method, params,
            jubacore::storage::storage_factory::create_storage("local")),
        jubafvconv::make_fv_converter(fvconv_config, NULL)));
    return handle;
}

int pthread_rwlockattr_destroy(pthread_rwlockattr_t *attr) {
    return 0;
}
int pthread_rwlockattr_init(pthread_rwlockattr_t *attr) {
    return 0;
}
int pthread_rwlock_rdlock(pthread_rwlock_t *rwlock) {
    return 0;
}
int pthread_rwlock_wrlock(pthread_rwlock_t *rwlock) {
    return 0;
}
int pthread_rwlock_unlock(pthread_rwlock_t *rwlock) {
    return 0;
}
int pthread_rwlock_destroy(pthread_rwlock_t *rwlock) {
    return 0;
}

void train(Handle *handle, int label, int count, float *values) {
    datum d;
    for (int i = 0; i < count; ++i) {
        d.num_values_.push_back(std::make_pair(std::to_string(i), values[i]));
        std::cout << "[DEBUG] train label=" << label << ". value=" << values[i] << std::endl;
    }
    handle->handle->train(std::to_string(label), d);
}

void classify(Handle *handle, int count, float *values, int *out_label, float *out_score) {
    datum d;
    for (int i = 0; i < count; ++i) {
        d.num_values_.push_back(std::make_pair(std::to_string(i), values[i]));
        std::cout << "[DEBUG] classify value=" << values[i] << std::endl;
    }
    auto ret = handle->handle->classify(d);
    std::cout << "[debug] returned " << ret.size() << " items" << std::endl;
    auto& top = ret[0];
    for (auto j = 1; j < ret.size(); ++j) {
        std::cout << "[debug] " << ret[j].label << ": " << ret[j].score << std::endl;
        if (top.score < ret[j].score)
            top = ret[j];
    }
    *out_label = std::atoi(top.label.c_str());
    *out_score = top.score;
}

}

int main() {
    const char *config_json_text = "{"
        "\"method\": \"PA\","
        "\"converter\": {"
            "\"num_filter_types\": {},"
            "\"num_filter_rules\": [],"
            "\"string_filter_types\": {},"
            "\"string_filter_rules\": [],"
            "\"num_types\": {},"
            "\"num_rules\": [{ \"key\": \"*\",  \"type\": \"num\" }],"
            "\"string_types\": {},"
            "\"string_rules\": ["
                "{ \"key\": \"*\", \"type\": \"space\", \"sample_weight\": \"bin\", \"global_weight\": \"bin\" }"
            "]"
        "},"
        "\"parameter\": {"
        "}"
    "}";
    auto handle = create_classifier(config_json_text);
    std::cout << "created classifier" << std::endl;
    float v;
    for (int i = 0; i < 10; ++i) {
        v = 1.0 * (i + 1);
        train(handle, 12345, 1, &v);
        v = -1.0 * (i + 1);
        train(handle, 54321, 1, &v);
    }
    int out_label;
    float out_score;
    v = 0.9;
    classify(handle, 1, &v, &out_label, &out_score);
    std::cout << out_label << ": " << out_score << std::endl;
    v = -1.1;
    classify(handle, 1, &v, &out_label, &out_score);
    std::cout << out_label << ": " << out_score << std::endl;
    delete handle;
    return 0;
}
