
#include "custom_fp.hh"

#include <iostream>
#include <omp.h>

#include "fp32.hh"

using namespace std;

void inc(torch::Tensor output, torch::Tensor input)
{
    auto val = input.accessor<float, 1>();
    auto res = output.accessor<float, 1>();

    fp32_t r, v1;
    v1.data = val[0];
    __fp32_inc(r, v1);
    res[0] = r.data;

    // res[0] = val[0] + 1;
}

void fp32_add(torch::Tensor result, torch::Tensor value1, torch::Tensor value2)
{
    auto val1 = value1.accessor<float, 1>();
    auto val2 = value2.accessor<float, 1>();
    auto res = result.accessor<float, 1>();

    fp32_t r, v1, v2;
    v1.data = val1[0];
    v2.data = val2[0];
    __fp32_add(r, v1, v2);
    res[0] = r.data;
}


/*
 *  My convolution
 */

void my_conv2d(torch::Tensor ofm,
               torch::Tensor ifm,
               torch::Tensor wgt,
               torch::Tensor bias,
               int stride_h,
               int stride_w,
               int pad_h,
               int pad_w,
               int dilation)
{
    omp_set_num_threads(4);
    // TODO: dilation

    auto ofm_a = ofm.accessor<float, 4>();
    auto ifm_a = ifm.accessor<float, 4>();
    auto wgt_a = wgt.accessor<float, 4>();
    auto bias_a = bias.accessor<float, 1>();

    const int batch = ifm_a.size(0);
    const int ifm_c = ifm_a.size(1);
    const int ifm_h = ifm_a.size(2);
    const int ifm_w = ifm_a.size(3);

    const int ofm_c = ofm_a.size(1);
    const int ofm_h = ofm_a.size(2);
    const int ofm_w = ofm_a.size(3);

    const int wgt_h = wgt_a.size(2);
    const int wgt_w = wgt_a.size(3);

    double start = omp_get_wtime();

# pragma omp parallel for collapse(4) default(none) shared(ifm_a, wgt_a, bias_a, ofm_a, stride_h, stride_w, pad_h, pad_w)
    for (int b = 0; b < batch; b ++) {
        for (int o_c = 0; o_c < ofm_c; o_c ++) {
            for (int o_h = 0; o_h < ofm_h; o_h ++) {
                for (int o_w = 0; o_w < ofm_w; o_w ++) {

                    float psum = 0.0;
                    for (int i_c = 0; i_c < ifm_c; i_c ++) {
                        for (int w_h = 0; w_h < wgt_h; w_h ++) {
                            for (int w_w = 0; w_w < wgt_w; w_w ++) {
                                int i_h = o_h * stride_h + w_h;
                                int i_w = o_w * stride_w + w_w;
                                if (i_h >= pad_h && i_h <= (pad_h + ifm_h) && i_w >= pad_w && i_w <= (pad_w + ifm_w)) {
                                    psum += ifm_a[b][i_c][i_h - pad_h][i_w - pad_w] * wgt_a[o_c][i_c][i_h][i_w];
                                }
                            }
                        }
                    }
                    ofm_a[b][o_c][o_h][o_w] = psum + bias_a[o_c];
                }
            }
        }
    }

    double end = omp_get_wtime();
    cout << "Elapsed time: " << (end - start) << " seconds" << endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("inc", &inc, "increase by 1");
    m.def("fp32_add", &fp32_add, "add two given numbers");
    m.def("my_conv2d", &my_conv2d, "2D convolution");
}

