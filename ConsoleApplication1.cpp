#include <iostream>
#include <Windows.h>
#include <sys/utime.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
//#include <arm_neon.h>
#include <immintrin.h>
using namespace std;
int N = 64;
void m_reset(float **m)
{
    for (int i = 0; i < N; i++)
    {
        m[i] = new float[N];
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
        {
            m[i][j] = 0;
        }
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
        {
            m[i][j] = rand();
        }
        for (int k = 0; k < N; k++)
        { 
            for (int i = k + 1; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    m[i][j] += m[k][j];
                }
            }
        }
    }
}
void LU(float** m)
{

    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void LU_AVX(float** m)
{
    for (int k = 0; k < N; k++)
    {

        __m128 vt = _mm_set1_ps(m[k][k]);
        for (int j = k + 1; j < N; j += 4)
        {
            if (j + 4 > N)
            {
                for (; j < N; j++)
                {
                    m[k][j] = m[k][j] / m[k][k];
                }
            }
            else
            {
                __m128 va = _mm_loadu_ps(m[k] + j);
                va = _mm_div_ps(va, vt);
                _mm_storeu_ps(m[k] + j, va);
            }
            m[k][k] = 1.0;
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k; j < N; j += 4)
            {
                if (j + 4 > N)
                {
                    for (; j < N; j++)
                    {
                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
                    }
                }
                else
                {
                    __m128 temp1 = _mm_loadu_ps(m[i] + j);
                    __m128 temp2 = _mm_loadu_ps(m[k] + j);
                    __m128 temp3 = _mm_set1_ps(m[i][k]);
                    temp2 = _mm_mul_ps(temp3, temp2);
                    temp1 = _mm_sub_ps(temp1, temp2);
                    _mm_storeu_ps(m[i] + j, temp1);
                }
                m[i][k] = 0;
            }
        }
    }
}
void LU_AVX_aligned(float** m)
{
    for (int k = 0; k < N; k++)
    {

        __m128 vt = _mm_set1_ps(m[k][k]);
        for (int j = k + 1; j < N; j += 4)
        {
            if (j + 4 > N)
            {
                for (; j < N; j++)
                {
                    m[k][j] = m[k][j] / m[k][k];
                }
            }
            else
            {
                __m128 va = _mm_load_ps(m[k] + j);
                va = _mm_div_ps(va, vt);
                _mm_store_ps(m[k] + j, va);
            }
            m[k][k] = 1.0;
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k; j < N; j += 4)
            {
                if (j + 4 > N)
                {
                    for (; j < N; j++)
                    {
                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
                    }
                }
                else
                {
                    __m128 temp1 = _mm_load_ps(m[i] + j);
                    __m128 temp2 = _mm_load_ps(m[k] + j);
                    __m128 temp3 = _mm_set1_ps(m[i][k]);
                    temp2 = _mm_mul_ps(temp3, temp2);
                    temp1 = _mm_sub_ps(temp1, temp2);
                    _mm_store_ps(m[i] + j, temp1);
                }
                m[i][k] = 0;
            }
        }
    }
}
void LU_AVX512(float** m)
{
    for (int k = 0; k < N; k++)
    {
        __m512 vt = _mm512_set1_ps(m[k][k]);
        for (int j = k + 1; j < N; j += 16)
        {
            if (j + 16 > N)
            {
                for (; j < N; j++)
                {
                    m[k][j] = m[k][j] / m[k][k];
                }
            }
            else
            {
                __m512 va = _mm512_loadu_ps(m[k] + j);
                va = _mm512_div_ps(va, vt);
                _mm512_storeu_ps(m[k] + j, va);
            }
            m[k][k] = 1.0;
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k; j < N; j += 16)
            {
                if (j + 16 > N)
                {
                    for (; j < N; j++)
                    {
                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
                    }
                }
                else
                {
                    __m512 temp1 = _mm512_loadu_ps(m[i] + j);
                    __m512 temp2 = _mm512_loadu_ps(m[k] + j);
                    __m512 temp3 = _mm512_set1_ps(m[i][k]);
                    temp2 = _mm512_mul_ps(temp3, temp2);
                    temp1 = _mm512_sub_ps(temp1, temp2);
                    _mm512_storeu_ps(m[i] + j, temp1);
                }
                m[i][k] = 0;
            }
        }
    }
}
//void LU_ARM_NEON()
//{
//    for (int k = 0; k < N; k++)
//    {
//        float32x4_t vt = vmovq_n_f32(m[k][k]);
//        for (int j = k + 1; j < N; j+=4)
//        {
//            if (j + 4 > N)
//            {
//                for (; j < N; j++)
//                {
//                    m[k][j] = m[k][j] / m[k][k];
//                }
//            }
//            else
//            {
//                float32x4_t va = vld1q_f32(m[k] + j);
//                va = vmulq_f32(va, vt);
//                vst1q_f32(m[k] + j, va);
//            }
//            m[k][k] = 1.0;
//        }
//        m[k][k] = 1.0;
//        for (int i = k + 1; i < N; i++)
//        {
//            for (int j = k; j < N; j+=4)
//            {
//                if (j + 4 > N)
//                {
//                    for (; j < N; j++)
//                    {
//                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
//                    }
//                }
//                else
//                {
//                    float32x4_t temp1 = vld1q_f32(m[i] + j);
//                    float32x4_t temp2 = vld1q_f32(m[k] + j);
//                    float32x4_t temp3 = vld1q_f32(m[i] + k);
//                    temp2 = vmulq_f32(temp3, temp2);
//                    temp1 = vsubq_f32(temp1, temp2);
//                    vst1q_f32(m[i] + j, temp1);
//                }
//                m[i][k] = 0;
//            }
//        }
//    }
//}
int main()
{

    float** m = new float* [N];
    m_reset(m);
    LU(m);

    //struct timespec sts, ets;
    //timespec_get(&sts, TIME_UTC);
    //for (int cycle = 0; cycle < 10; cycle++)
    //{
    //    LU_AVX512(m);
    //}

    //timespec_get(&ets, TIME_UTC);
    //time_t dsec = ets.tv_sec - sts.tv_sec;
    //long dnsec = ets.tv_nsec - sts.tv_nsec;
    //if (dnsec < 0)
    //{
    //    dsec--;
    //    dnsec+=1000000000ll;
    //}
    //printf("%lld.%09llds\n",dsec,dnsec);


    //struct timespec sts2, ets2;
    //timespec_get(&sts2, TIME_UTC);
    //for (int cycle = 0; cycle < 1; cycle++)
    //{
    //    LU_AVX();
    //}

    //timespec_get(&ets2, TIME_UTC);
    //time_t dsec2 = ets2.tv_sec - sts2.tv_sec;
    //long dnsec2 = ets2.tv_nsec - sts2.tv_nsec;
    //if (dnsec2 < 0)
    //{
    //    dsec2--;
    //    dnsec2 += 1000000000ll;
    //}
    //printf("%lld.%09llds\n", dsec2, dnsec2);


    //struct timespec sts3, ets3;
    //timespec_get(&sts3, TIME_UTC);
    //for (int cycle = 0; cycle < 1; cycle++)
    //{
    //    LU_AVX_aligned();
    //}

    //timespec_get(&ets3, TIME_UTC);
    //time_t dsec3 = ets3.tv_sec - sts3.tv_sec;
    //long dnsec3 = ets3.tv_nsec - sts3.tv_nsec;
    //if (dnsec3 < 0)
    //{
    //    dsec3--;
    //    dnsec3 += 1000000000ll;
    //}
    //printf("%lld.%09llds\n", dsec3, dnsec3);

    return 0;
}


