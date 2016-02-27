#if defined(USE_OPENCL) && !defined(JUBATUS_UTIL_OPENCL_HPP_)
#define JUBATUS_UTIL_OPENCL_HPP_

#include <string>
#include <CL/opencl.h>

namespace jubatus {
namespace util {
namespace cl {

class ContextAndQueue {
public:
  ContextAndQueue();
  ~ContextAndQueue();
  inline const cl_context context() const { return context_; }
  inline const cl_device_id device_id() const { return device_id_; }
  inline const cl_command_queue queue() const { return queue_; }
private:
  cl_platform_id platform_id_;
  cl_device_id device_id_;
  cl_context context_;
  cl_command_queue queue_;
};

class Program {
public:
  Program(const ContextAndQueue& cq);
  ~Program();

  void build(const std::string& code, const std::string& options);
  cl_kernel get(const std::string& name) const;

  inline const cl_context context() const { return context_.context(); }
  inline const cl_device_id device_id() const { return context_.device_id(); }
  inline const cl_command_queue queue() const { return context_.queue(); }
private:
  const ContextAndQueue& context_;
  cl_program program_;
};

extern const ContextAndQueue GPU;

}}}

#endif
