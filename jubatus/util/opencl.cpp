#ifdef USE_OPENCL
#include <iostream>
#include <stdexcept>
#include "opencl.hpp"

namespace jubatus {
namespace util {
namespace cl {

const ContextAndQueue GPU;

ContextAndQueue::ContextAndQueue()
  : platform_id_(NULL), device_id_(NULL), context_(NULL), queue_(NULL)
{
  if (clGetPlatformIDs(1, &platform_id_, NULL) != CL_SUCCESS)
    throw std::runtime_error("failed clGetPlatformIDs");
  if (clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, 1, &device_id_, NULL) != CL_SUCCESS)
    throw std::runtime_error("OpenCL Device NotFound");
  context_ = clCreateContext(0, 1, &device_id_, NULL, NULL, NULL);
  if (!context_)
    throw std::runtime_error("failed clCreateContext");
  queue_ = clCreateCommandQueueWithProperties(context_, device_id_, 0, NULL);
  if (!queue_)
    throw std::runtime_error("failed clCreateCommandQueue");
  std::cout << "Get OpenCL Device" << std::endl;
}

ContextAndQueue::~ContextAndQueue()
{
  if (queue_)
    clReleaseCommandQueue(queue_);
  if (context_)
    clReleaseContext(context_);
  std::cout << "Release OpenCL Device" << std::endl;
}

Program::Program(const ContextAndQueue& cq) : context_(cq), program_(NULL)
{
}

Program::~Program()
{
  if (program_)
    clReleaseProgram(program_);
}

void Program::build(const std::string& code, const std::string& options) {
  const char *p = code.c_str();
  size_t len = code.size();
  program_ = clCreateProgramWithSource(context(), 1, &p, &len, NULL);
  if (!program_)
    throw std::runtime_error("failed clCreateProgramWithSource");

  auto did = device_id();
  if (clBuildProgram(program_, 1, &did, options.c_str(), NULL, NULL) != CL_SUCCESS)
    throw std::runtime_error("failed clBuildProgram");
}

cl_kernel Program::get(const std::string& name) const {
  return clCreateKernel(program_, name.c_str(), NULL);
}
  

}}}

#endif
