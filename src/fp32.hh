#ifndef __FP32_HH__
#define __FP32_HH__

#include <torch/extension.h>

typedef union {
    float data;
    uint32_t raw;
    struct {
        uint32_t fraction : 23;
        uint32_t exponent : 8;
        uint32_t sign : 1;
    } bits;
} fp32_t;

void __fp32_inc(fp32_t& res, fp32_t& val);
void __fp32_add(fp32_t& res, fp32_t& val1, fp32_t& val2);

#endif // __FP32_HH__
