#ifndef __CUSTOM_FP_HH__
#define __CUSTOM_FP_HH__

#include <torch/extension.h>

void inc(torch::Tensor ret, torch::Tensor input);
void fp32_add(torch::Tensor result, torch::Tensor value1, torch::Tensor value2);

#endif // __CUSTOM_FP_HH__
