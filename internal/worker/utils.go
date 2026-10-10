package worker

import "os"

// 判断文件是否存在
func PathExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, err
	}
	return false, err
}

// 对数组去重
func unique(labels []uint32) []uint32 {
	uniqueLabels := make([]uint32, 0)
	temp := make(map[uint32]struct{})
	for _, label := range labels {
		if _, ok := temp[label]; !ok {
			temp[label] = struct{}{}
			uniqueLabels = append(uniqueLabels, label)
		}
	}
	return uniqueLabels
}
