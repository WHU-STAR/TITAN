//go:build arm64

#include "textflag.h"

// l2DistanceNEON 使用ARM NEON指令计算L2距离
// func l2DistanceNEON(a, b unsafe.Pointer, dim int) float32
TEXT ·l2DistanceNEON(SB), NOSPLIT, $0-24
	MOVD a+0(FP), R0    // 向量a的指针
	MOVD b+8(FP), R1    // 向量b的指针
	MOVD dim+16(FP), R2 // 维度
	
	// 初始化累加器为零
	MOVI V0.4S, #0      // V0 = 0 (128位零向量)
	
	// 主循环：每次处理4个float32
	MOVD R2, R3
	LSR  $2, R3         // R3 = dim / 4
	CBZ  R3, remainder_neon
	
loop_neon:
	// 加载4个float32
	LDP  (R0), (V1.D, V2.D)  // V1 = a[i:i+4]
	LDP  (R1), (V3.D, V4.D)  // V3 = b[i:i+4]
	
	// 计算差值
	FSUB V1.4S, V3.4S, V5    // V5 = b[i:i+4] - a[i:i+4]
	
	// 计算平方并累加
	FMLA V0.4S, V5.4S, V5.4S // V0 += V5 * V5
	
	// 更新指针
	ADD  $16, R0        // 4 * 4 bytes
	ADD  $16, R1
	SUB  $1, R3
	CBNZ R3, loop_neon
	
remainder_neon:
	// 处理剩余元素
	AND  $3, R2         // R2 = dim % 4
	CBZ  R2, reduce_neon
	
	// 使用标量处理剩余元素
	MOVI V6.4S, #0      // V6 = 0
remainder_loop_neon:
	LDR  S7, (R0)       // S7 = a[i]
	LDR  S8, (R1)       // S8 = b[i]
	FSUB S7, S8, S9     // S9 = b[i] - a[i]
	FMUL S9, S9, S9     // S9 = (b[i] - a[i])^2
	FADD S6, S9, S6     // S6 += (b[i] - a[i])^2
	
	ADD  $4, R0
	ADD  $4, R1
	SUB  $1, R2
	CBNZ R2, remainder_loop_neon
	
	// 将标量结果加到向量累加器中
	FADD V0.4S, V6.4S, V0
	
reduce_neon:
	// 水平求和：将4个float32求和为1个
	FADDP V0.4S, V0.4S, V0    // 成对加法
	FADDP V0.2S, V0.2S, V0    // 再次成对加法
	
	// 计算平方根
	FSQRT S0, S0              // S0 = sqrt(S0)
	
	// 返回结果
	FMOV S0, R0
	MOVD R0, ret+24(FP)
	RET
