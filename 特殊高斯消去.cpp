#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <ctime>
#include <cmath>
//#include <arm_neon.h>
#include <immintrin.h>
using namespace std;

#define COL 3799
#define Eliminator 2759
#define ROW 1953
// Global arrays for storing data
unsigned int eliminator[COL][COL / 32 + 1] = { {0} };
unsigned int row[ROW][COL / 32 + 1] = { {0} };
unsigned int eli_temp[COL][COL / 32 + 1] = { {0} };
unsigned int row_temp[ROW][COL / 32 + 1] = { {0} };

void groebner_serial(unsigned int eliminator[COL][COL / 32 + 1], unsigned int row[ROW][COL / 32 + 1]) 
{
    memcpy(eli_temp, eliminator, sizeof(unsigned int) * COL * (COL / 32 + 1));
    memcpy(row_temp, row, sizeof(unsigned int) * ROW * (COL / 32 + 1));
    for (int i = 0; i < ROW; i++) {
        for (int j = COL; j >= 0; j--) {
            if (row_temp[i][j / 32] & ((unsigned int)1 << (j % 32))) {
                if (eli_temp[j][j / 32] & ((unsigned int)1 << (j % 32))) {
                    for (int p = COL / 32; p >= 0; p--)
                        row_temp[i][p] ^= eli_temp[j][p];
                }
                else {
                    memcpy(eli_temp[j], row_temp[i], (COL / 32 + 1) * sizeof(unsigned int));
                    break;
                }
            }
        }
    }
}
void groebner_sse(unsigned int eliminator[COL][COL / 32 + 1], unsigned int row[ROW][COL / 32 + 1]) 
{

    memcpy(eli_temp, eliminator, sizeof(unsigned int) * COL * (COL / 32 + 1));
    memcpy(row_temp, row, sizeof(unsigned int) * ROW * (COL / 32 + 1));
    __m128i row_i, eli_j;
    for (int i = 0; i < ROW; i++) {
        for (int j = COL; j >= 0; j--) {
            if (row_temp[i][j / 32] & ((unsigned int)1 << (j % 32))) {
                if (eli_temp[j][j / 32] & ((unsigned int)1 << (j % 32))) {
                    for (int p = 0; p < COL / 128; p++) {
                        row_i = _mm_loadu_si128((__m128i*)(row_temp[i] + p * 4));
                        eli_j = _mm_loadu_si128((__m128i*)(eli_temp[j] + p * 4));
                        _mm_storeu_si128((__m128i*)(row_temp[i] + p * 4), _mm_xor_si128(row_i, eli_j));
                    }
                    for (int k = COL / 128 * 4; k <= COL / 32; k++)
                        row_temp[i][k] ^= eli_temp[j][k];
                }
                else {
                    memcpy(eli_temp[j], row_temp[i], (COL / 32 + 1) * sizeof(unsigned int));
                    break;
                }
            }
        }
    }

}
void groebner_avx(unsigned int eliminator[COL][COL / 32 + 1], unsigned int row[ROW][COL / 32 + 1]) 
{
    memcpy(eli_temp, eliminator, sizeof(unsigned int) * COL * (COL / 32 + 1));
    memcpy(row_temp, row, sizeof(unsigned int) * ROW * (COL / 32 + 1));
    __m256i row_i, eli_j;
    for (int i = 0; i < ROW; i++) {
        for (int j = COL; j >= 0; j--) {
            if (row_temp[i][j / 32] & ((unsigned int)1 << (j % 32))) {
                if (eli_temp[j][j / 32] & ((unsigned int)1 << (j % 32))) {
                    for (int p = 0; p < COL / 256; p++) {
#ifdef aligned
                        row_i = _mm256_load_si256((__m256i*)(row_temp[i] + p * 8));
                        eli_j = _mm256_load_si256((__m256i*)(eli_temp[j] + p * 8));
                        _mm256_store_si256((__m256i*)(row_temp[i] + p * 8), _mm256_xor_si256(row_i, eli_j));
#else
                        row_i = _mm256_loadu_si256((__m256i*)(row_temp[i] + p * 8));
                        eli_j = _mm256_loadu_si256((__m256i*)(eli_temp[j] + p * 8));
                        _mm256_storeu_si256((__m256i*)(row_temp[i] + p * 8), _mm256_xor_si256(row_i, eli_j));
#endif
                    }
                    for (int k = COL / 256 * 8; k <= COL / 32; k++)
                        row_temp[i][k] ^= eli_temp[j][k];
                }
                else {
                    memcpy(eli_temp[j], row_temp[i], (COL / 32 + 1) * sizeof(unsigned int));
                    break;
                }
            }
        }
    }
}

void groebner_avx512(unsigned int eliminator[COL][COL / 32 + 1], unsigned int row[ROW][COL / 32 + 1]) 
{
    memcpy(eli_temp, eliminator, sizeof(unsigned int) * COL * (COL / 32 + 1));
    memcpy(row_temp, row, sizeof(unsigned int) * ROW * (COL / 32 + 1));
    __m512i row_i, eli_j;
    for (int i = 0; i < ROW; i++) {
        for (int j = COL; j >= 0; j--) {
            if (row_temp[i][j / 32] & ((unsigned int)1 << (j % 32))) {
                if (eli_temp[j][j / 32] & ((unsigned int)1 << (j % 32))) {
                    for (int p = 0; p < COL / 512; p++) {
#ifdef aligned
                        row_i = _mm512_load_si512((__m512i*)(row_temp[i] + p * 16));
                        eli_j = _mm512_load_si512((__m512i*)(eli_temp[j] + p * 16));
                        _mm512_store_si512((__m512i*)(row_temp[i] + p * 16), _mm512_xor_si512(row_i, eli_j));
#else
                        row_i = _mm512_loadu_si512((__m512i*)(row_temp[i] + p * 16));
                        eli_j = _mm512_loadu_si512((__m512i*)(eli_temp[j] + p * 16));
                        _mm512_storeu_si512((__m512i*)(row_temp[i] + p * 16), _mm512_xor_si512(row_i, eli_j));
#endif
                    }
                    for (int k = COL / 512 * 16; k <= COL / 32; k++)
                        row_temp[i][k] ^= eli_temp[j][k];
                }
                else {
                    memcpy(eli_temp[j], row_temp[i], (COL / 32 + 1) * sizeof(unsigned int));
                    break;
                }
            }
        }
    }

}


//获取计算时间
void get_time(void (*func)(unsigned int[COL][COL / 32 + 1], unsigned int[ROW][COL / 32 + 1]), const char* msg) {
    struct timespec sts, ets;
    timespec_get(&sts, TIME_UTC);
    for (int cycle = 0; cycle < 10; cycle++)
    {
        func(eliminator, row);
    }

    timespec_get(&ets, TIME_UTC);
    time_t dsec = ets.tv_sec - sts.tv_sec;
    long dnsec = ets.tv_nsec - sts.tv_nsec;
    if (dnsec < 0)
    {
        dsec--;
        dnsec += 1000000000ll;
    }
    printf("%lld.%09llds\n", dsec, dnsec);
}

int main() {
    ifstream data_eli("./groebner/dataset/data1.txt", ios::in);
    int temp, header;
    string line;
    for (int i = 0; i < Eliminator; i++) {
        getline(data_eli, line);
        istringstream Get_line(line);
        Get_line >> header;
        eliminator[header][header / 32] += (unsigned int)1 << (header % 32);
        while (Get_line >> temp)
            eliminator[header][temp / 32] += (unsigned int)1 << (temp % 32);
    }
    data_eli.close();


    get_time(groebner_serial, "serial");

    return 0;
}
