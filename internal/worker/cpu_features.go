package worker

import (
	"runtime"
	"strings"
)

// CPUFeatures 存储CPU特性信息
type CPUFeatures struct {
	AVX2 bool
	SSE2 bool
	NEON bool
	FMA  bool
	AVX  bool
}

var cpuFeatures *CPUFeatures

// GetCPUFeatures 获取当前CPU的特性
func GetCPUFeatures() *CPUFeatures {
	if cpuFeatures == nil {
		cpuFeatures = detectCPUFeatures()
	}
	return cpuFeatures
}

// detectCPUFeatures 检测CPU特性
func detectCPUFeatures() *CPUFeatures {
	features := &CPUFeatures{}

	// 根据架构设置默认特性
	switch runtime.GOARCH {
	case "amd64":
		features.SSE2 = true // SSE2在x86_64上是标准特性
		features.AVX = detectAVX()
		features.AVX2 = detectAVX2()
		features.FMA = detectFMA()
	case "arm64":
		features.NEON = true // NEON在ARM64上是标准特性
	}

	return features
}

// detectAVX 检测AVX支持
func detectAVX() bool {
	// 实际实现中应该使用cpuid指令
	// 这里简化处理，在x86_64平台上假设支持
	return runtime.GOARCH == "amd64"
}

// detectAVX2 检测AVX2支持
func detectAVX2() bool {
	// 实际实现中应该使用cpuid指令
	// 这里简化处理，在x86_64平台上假设支持
	return runtime.GOARCH == "amd64"
}

// detectFMA 检测FMA支持
func detectFMA() bool {
	// 实际实现中应该使用cpuid指令
	// 这里简化处理，在x86_64平台上假设支持
	return runtime.GOARCH == "amd64"
}

// hasAVX2 检查是否支持AVX2
func hasAVX2() bool {
	return GetCPUFeatures().AVX2
}

// hasSSE2 检查是否支持SSE2
func hasSSE2() bool {
	return GetCPUFeatures().SSE2
}

// hasNEON 检查是否支持NEON
func hasNEON() bool {
	return GetCPUFeatures().NEON
}

// GetOptimalSIMDType 获取最优的SIMD类型
func GetOptimalSIMDType() string {
	features := GetCPUFeatures()

	switch runtime.GOARCH {
	case "amd64":
		if features.AVX2 {
			return "AVX2"
		}
		if features.AVX {
			return "AVX"
		}
		if features.SSE2 {
			return "SSE2"
		}
	case "arm64":
		if features.NEON {
			return "NEON"
		}
	}

	return "SCALAR"
}

// PrintCPUInfo 打印CPU信息
func PrintCPUInfo() string {
	features := GetCPUFeatures()
	arch := runtime.GOARCH
	os := runtime.GOOS

	var info strings.Builder
	info.WriteString("CPU架构: " + arch + "\n")
	info.WriteString("操作系统: " + os + "\n")
	info.WriteString("最优SIMD类型: " + GetOptimalSIMDType() + "\n")

	info.WriteString("支持的SIMD指令集:\n")
	if features.AVX2 {
		info.WriteString("  ✓ AVX2\n")
	}
	if features.AVX {
		info.WriteString("  ✓ AVX\n")
	}
	if features.SSE2 {
		info.WriteString("  ✓ SSE2\n")
	}
	if features.FMA {
		info.WriteString("  ✓ FMA\n")
	}
	if features.NEON {
		info.WriteString("  ✓ NEON\n")
	}

	return info.String()
}
