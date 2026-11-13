// Basin inference-only RWKV-7 recurrence. Equations and accumulation order follow
// BlinkDL/RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_clampw.cu (vendored backend).
// State is [batch, head, value, key], float32; activations are bfloat16.
#include <cuda_bf16.h>

__global__ void recurrence(const __nv_bfloat16* r, const __nv_bfloat16* w,
    const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* a,
    const __nv_bfloat16* b, __nv_bfloat16* y, float* state, int T, int H) {
    int h = blockIdx.x, batch = blockIdx.y, i = threadIdx.x;
    int base = ((batch * H + h) * 64 + i) * 64;
    float s[64];
    #pragma unroll
    for (int j = 0; j < 64; ++j) s[j] = state[base + j];
    __shared__ float rr[16][64], ww[16][64], kk[16][64], aa[16][64], bb[16][64], vv[16][64];
    for (int t0 = 0; t0 < T; t0 += 16) {
        __syncthreads();
        #pragma unroll
        for (int tt = 0; tt < 16; ++tt) {
            if (t0 + tt < T) {
                int idx = ((batch * T + t0 + tt) * H + h) * 64 + i;
                rr[tt][i] = __bfloat162float(r[idx]);
                ww[tt][i] = __expf(-0.6065306597f / (1.0f + __expf(-__bfloat162float(w[idx]))));
                kk[tt][i] = __bfloat162float(k[idx]);
                aa[tt][i] = __bfloat162float(a[idx]);
                bb[tt][i] = __bfloat162float(b[idx]);
                vv[tt][i] = __bfloat162float(v[idx]);
            }
        }
        __syncthreads();
        for (int tt = 0; tt < 16 && t0 + tt < T; ++tt) {
            float sa = 0.0f, out = 0.0f, vi = vv[tt][i];
            #pragma unroll
            for (int j = 0; j < 64; ++j) sa += s[j] * aa[tt][j];
            #pragma unroll
            for (int j = 0; j < 64; ++j) {
                s[j] = s[j] * ww[tt][j] + (sa * bb[tt][j] + kk[tt][j] * vi);
                out += s[j] * rr[tt][j];
            }
            int idx = ((batch * T + t0 + tt) * H + h) * 64 + i;
            y[idx] = __float2bfloat16_rn(out);
        }
    }
    #pragma unroll
    for (int j = 0; j < 64; ++j) state[base + j] = s[j];
}

void launch(const void* r, const void* w, const void* k, const void* v, const void* a, const void* b, void* y, float* state, int B, int T, int H, cudaStream_t stream) {
    recurrence<<<dim3(H,B),64,0,stream>>>((const __nv_bfloat16*)r, (const __nv_bfloat16*)w,
        (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, (const __nv_bfloat16*)a,
        (const __nv_bfloat16*)b, (__nv_bfloat16*)y, state, T, H);
}
