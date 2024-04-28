#include <iostream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>
using namespace std;
const int N = 200;
float m[N][N];
void m_reset()
{
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
void LU_ARM_NEON()
{
   for (int k = 0; k < N; k++)
   {
       float32x4_t vt = vmovq_n_f32(m[k][k]);
       for (int j = k + 1; j < N; j+=4)
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
               float32x4_t va = vld1q_f32(m[k] + j);
               va = vmulq_f32(va, vt);
               vst1q_f32(m[k] + j, va);
           }
           m[k][k] = 1.0;
       }
       m[k][k] = 1.0;
       for (int i = k + 1; i < N; i++)
       {
           for (int j = k; j < N; j+=4)
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
                   float32x4_t temp1 = vld1q_f32(m[i] + j);
                   float32x4_t temp2 = vld1q_f32(m[k] + j);
                   float32x4_t temp3 = vld1q_f32(m[i] + k);
                   temp2 = vmulq_f32(temp3, temp2);
                   temp1 = vsubq_f32(temp1, temp2);
                   vst1q_f32(m[i] + j, temp1);
               }
               m[i][k] = 0;
           }
       }
   }
}
int main()
{
    m_reset();
    struct timespec sts, ets;
    timespec_get(&sts, TIME_UTC);
    for (int cycle = 0; cycle < 10; cycle++)
    {
        LU_ARM_NEON();
    }

    timespec_get(&ets, TIME_UTC);
    time_t dsec = ets.tv_sec - sts.tv_sec;
    long dnsec = ets.tv_nsec - sts.tv_nsec;
    if (dnsec < 0)
    {
        dsec--;
        dnsec+=1000000000ll;
    }
    printf("%lld.%09llds\n",dsec,dnsec);
    cout << endl;


}