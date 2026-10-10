//go:build amd64

#include "textflag.h"

// l2DistanceAVX2 使用AVX2指令计算L2距离
// func l2DistanceAVX2(a, b unsafe.Pointer, dim int) float32
TEXT ·l2DistanceAVX2(SB), NOSPLIT, $0-24
	MOVQ a+0(FP), AX    // 向量a的指针
	MOVQ b+8(FP), BX    // 向量b的指针  
	MOVQ dim+16(FP), CX // 维度
	
	// 初始化累加器为零
	VXORPS Y0, Y0, Y0   // Y0 = 0 (256位零向量)
	
	// 主循环：每次处理8个float32
	MOVQ CX, DX
	SHRQ $3, DX         // DX = dim / 8
	JZ   remainder_avx2
	
loop_avx2:
	// 加载8个float32
	VMOVUPS (AX), Y1    // Y1 = a[i:i+8]
	VMOVUPS (BX), Y2    // Y2 = b[i:i+8]
	
	// 计算差值
	VSUBPS Y1, Y2, Y3   // Y3 = b[i:i+8] - a[i:i+8]
	
	// 计算平方并累加
	VFMADD231PS Y3, Y3, Y0  // Y0 += Y3 * Y3
	
	// 更新指针
	ADDQ $32, AX        // 8 * 4 bytes
	ADDQ $32, BX
	DECQ DX
	JNZ  loop_avx2
	
remainder_avx2:
	// 处理剩余元素
	ANDQ $7, CX         // CX = dim % 8
	JZ   reduce_avx2
	
	// 使用标量处理剩余元素
	XORPS X4, X4        // X4 = 0
remainder_loop_avx2:
	MOVSS (AX), X5      // X5 = a[i]
	SUBSS (BX), X5      // X5 = a[i] - b[i]
	MULSS X5, X5        // X5 = (a[i] - b[i])^2
	ADDSS X5, X4        // X4 += (a[i] - b[i])^2
	
	ADDQ $4, AX
	ADDQ $4, BX
	DECQ CX
	JNZ  remainder_loop_avx2
	
	// 将标量结果加到向量累加器中
	VADDSS X4, X0, X0
	
reduce_avx2:
	// 水平求和：将8个float32求和为1个
	VEXTRACTF128 $1, Y0, X1  // X1 = Y0[128:256]
	VADDPS X0, X1, X0        // X0 = X0 + X1
	VHADDPS X0, X0, X0       // 水平加法
	VHADDPS X0, X0, X0       // 再次水平加法
	
	// 计算平方根
	VSQRTSS X0, X0, X0       // X0 = sqrt(X0)
	
	// 返回结果
	MOVSS X0, ret+24(FP)
	RET

// l2DistanceSquaredSSE2 使用SSE2指令计算L2距离平方（不计算sqrt）
// func l2DistanceSquaredSSE2(a, b unsafe.Pointer, dim int) float32
TEXT ·l2DistanceSquaredSSE2(SB), NOSPLIT, $0-24
	MOVQ a+0(FP), AX    // 向量a的指针
	MOVQ b+8(FP), BX    // 向量b的指针
	MOVQ dim+16(FP), CX // 维度
	
	// 初始化累加器为零
	XORPS X0, X0        // X0 = 0
	
	// 主循环：每次处理4个float32
	MOVQ CX, DX
	SHRQ $2, DX         // DX = dim / 4
	JZ   remainder_sq_sse2
	
loop_sq_sse2:
	// 加载4个float32
	MOVUPS (AX), X1     // X1 = a[i:i+4]
	MOVUPS (BX), X2     // X2 = b[i:i+4]
	
	// 计算差值
	SUBPS X1, X2        // X2 = b[i:i+4] - a[i:i+4]
	
	// 计算平方并累加
	MULPS X2, X2        // X2 = X2 * X2
	ADDPS X2, X0        // X0 += X2
	
	// 更新指针
	ADDQ $16, AX        // 4 * 4 bytes
	ADDQ $16, BX
	DECQ DX
	JNZ  loop_sq_sse2
	
remainder_sq_sse2:
	// 处理剩余元素
	ANDQ $3, CX         // CX = dim % 4
	JZ   reduce_sq_sse2
	
remainder_sq_loop_sse2:
	MOVSS (AX), X1      // X1 = a[i]
	SUBSS (BX), X1      // X1 = a[i] - b[i]
	MULSS X1, X1        // X1 = (a[i] - b[i])^2
	ADDSS X1, X0        // X0 += (a[i] - b[i])^2
	
	ADDQ $4, AX
	ADDQ $4, BX
	DECQ CX
	JNZ  remainder_sq_loop_sse2
	
reduce_sq_sse2:
	// 水平求和：将4个float32求和为1个（不计算sqrt）
	HADDPS X0, X0       // 水平加法
	HADDPS X0, X0       // 再次水平加法
	
	// 返回结果（不计算sqrt）
	MOVSS X0, ret+24(FP)
	RET
// func l2DistanceSquaredAVX2(a, b unsafe.Pointer, dim int) float32
TEXT ·l2DistanceSquaredAVX2(SB), NOSPLIT, $0-24
	MOVQ a+0(FP), AX    // 向量a的指针
	MOVQ b+8(FP), BX    // 向量b的指针  
	MOVQ dim+16(FP), CX // 维度
	
	// 初始化累加器为零
	VXORPS Y0, Y0, Y0   // Y0 = 0 (256位零向量)
	
	// 主循环：每次处理8个float32
	MOVQ CX, DX
	SHRQ $3, DX         // DX = dim / 8
	JZ   remainder_sq_avx2
	
loop_sq_avx2:
	// 加载8个float32
	VMOVUPS (AX), Y1    // Y1 = a[i:i+8]
	VMOVUPS (BX), Y2    // Y2 = b[i:i+8]
	
	// 计算差值
	VSUBPS Y1, Y2, Y3   // Y3 = b[i:i+8] - a[i:i+8]
	
	// 计算平方并累加
	VFMADD231PS Y3, Y3, Y0  // Y0 += Y3 * Y3
	
	// 更新指针
	ADDQ $32, AX        // 8 * 4 bytes
	ADDQ $32, BX
	DECQ DX
	JNZ  loop_sq_avx2
	
remainder_sq_avx2:
	// 处理剩余元素
	ANDQ $7, CX         // CX = dim % 8
	JZ   reduce_sq_avx2
	
	// 使用标量处理剩余元素
	XORPS X4, X4        // X4 = 0
remainder_sq_loop_avx2:
	MOVSS (AX), X5      // X5 = a[i]
	SUBSS (BX), X5      // X5 = a[i] - b[i]
	MULSS X5, X5        // X5 = (a[i] - b[i])^2
	ADDSS X5, X4        // X4 += (a[i] - b[i])^2
	
	ADDQ $4, AX
	ADDQ $4, BX
	DECQ CX
	JNZ  remainder_sq_loop_avx2
	
	// 将标量结果加到向量累加器中
	VADDSS X4, X0, X0
	
reduce_sq_avx2:
	// 水平求和：将8个float32求和为1个（不计算sqrt）
	VEXTRACTF128 $1, Y0, X1  // X1 = Y0[128:256]
	VADDPS X0, X1, X0        // X0 = X0 + X1
	VHADDPS X0, X0, X0       // 水平加法
	VHADDPS X0, X0, X0       // 再次水平加法
	
	// 返回结果（不计算sqrt）
	MOVSS X0, ret+24(FP)
	RET

// l2DistanceSSE2 使用SSE2指令计算L2距离
// func l2DistanceSSE2(a, b unsafe.Pointer, dim int) float32
TEXT ·l2DistanceSSE2(SB), NOSPLIT, $0-24
	MOVQ a+0(FP), AX    // 向量a的指针
	MOVQ b+8(FP), BX    // 向量b的指针
	MOVQ dim+16(FP), CX // 维度
	
	// 初始化累加器为零
	XORPS X0, X0        // X0 = 0
	
	// 主循环：每次处理4个float32
	MOVQ CX, DX
	SHRQ $2, DX         // DX = dim / 4
	JZ   remainder_sse2
	
loop_sse2:
	// 加载4个float32
	MOVUPS (AX), X1     // X1 = a[i:i+4]
	MOVUPS (BX), X2     // X2 = b[i:i+4]
	
	// 计算差值
	SUBPS X1, X2        // X2 = b[i:i+4] - a[i:i+4]
	
	// 计算平方并累加
	MULPS X2, X2        // X2 = X2 * X2
	ADDPS X2, X0        // X0 += X2
	
	// 更新指针
	ADDQ $16, AX        // 4 * 4 bytes
	ADDQ $16, BX
	DECQ DX
	JNZ  loop_sse2
	
remainder_sse2:
	// 处理剩余元素
	ANDQ $3, CX         // CX = dim % 4
	JZ   reduce_sse2
	
	// 使用标量处理剩余元素
	XORPS X3, X3        // X3 = 0
remainder_loop_sse2:
	MOVSS (AX), X4      // X4 = a[i]
	SUBSS (BX), X4      // X4 = a[i] - b[i]
	MULSS X4, X4        // X4 = (a[i] - b[i])^2
	ADDSS X4, X3        // X3 += (a[i] - b[i])^2
	
	ADDQ $4, AX
	ADDQ $4, BX
	DECQ CX
	JNZ  remainder_loop_sse2
	
	// 将标量结果加到向量累加器中
	ADDSS X3, X0
	
reduce_sse2:
	// 水平求和：将4个float32求和为1个
	HADDPS X0, X0       // 水平加法
	HADDPS X0, X0       // 再次水平加法
	
	// 计算平方根
	SQRTSS X0, X0       // X0 = sqrt(X0)
	
	// 返回结果
	MOVSS X0, ret+24(FP)
	RET
