#include "fp32.hh"

#include <iostream>

using namespace std;

void __fp32_inc(fp32_t& res, fp32_t& val)
{
    res.data = val.data + 1.0;
}

void __fp32_add(fp32_t& res, fp32_t& val1, fp32_t& val2)
{
    cout << val1.data << ", " << val2.data << endl;
    res.data = val1.data + val2.data;
}
