package worker

import (
	"runtime"
	"strings"
)

// CPUFeatures records SIMD features.
type CPUFeatures struct {
	AVX2 bool
	SSE2 bool
	NEON bool
	FMA  bool
	AVX  bool
}

var cpuFeatures *CPUFeatures

// GetCPUFeatures returns the detected features.
func GetCPUFeatures() *CPUFeatures {
	if cpuFeatures == nil {
		cpuFeatures = detectCPUFeatures()
	}
	return cpuFeatures
}

// detectCPUFeatures fills the feature struct.
func detectCPUFeatures() *CPUFeatures {
	features := &CPUFeatures{}

	// Architecture defaults.
	switch runtime.GOARCH {
	case "amd64":
		features.SSE2 = true // required on x86_64
		features.AVX = detectAVX()
		features.AVX2 = detectAVX2()
		features.FMA = detectFMA()
	case "arm64":
		features.NEON = true // required on arm64
	}

	return features
}

// detectAVX reports AVX support.
func detectAVX() bool {
	// A production build should query CPUID.
	// Assume support on x86_64.
	return runtime.GOARCH == "amd64"
}

// detectAVX2 reports AVX2 support.
func detectAVX2() bool {
	// A production build should query CPUID.
	// Assume support on x86_64.
	return runtime.GOARCH == "amd64"
}

// detectFMA reports FMA support.
func detectFMA() bool {
	// A production build should query CPUID.
	// Assume support on x86_64.
	return runtime.GOARCH == "amd64"
}

// hasAVX2 reports AVX2.
func hasAVX2() bool {
	return GetCPUFeatures().AVX2
}

// hasSSE2 reports SSE2.
func hasSSE2() bool {
	return GetCPUFeatures().SSE2
}

// hasNEON reports NEON.
func hasNEON() bool {
	return GetCPUFeatures().NEON
}

// GetOptimalSIMDType returns the preferred SIMD path.
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

// PrintCPUInfo formats a CPU summary.
func PrintCPUInfo() string {
	features := GetCPUFeatures()
	arch := runtime.GOARCH
	os := runtime.GOOS

	var info strings.Builder
	info.WriteString("arch: " + arch + "\n")
	info.WriteString("os: " + os + "\n")
	info.WriteString("simd: " + GetOptimalSIMDType() + "\n")

	info.WriteString("instruction sets:\n")
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
