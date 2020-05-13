
#include <torch/extension.h>

#include <iostream>

using namespace std;

void inc(torch::Tensor ret, torch::Tensor input)
{
    auto val = input.accessor<long, 1>();
    ret[0] = val[0] + 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("inc", &inc, "increase by 1");
}
