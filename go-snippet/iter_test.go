package snippet

import (
	"fmt"
	"slices"
	"testing"
)

func TestIter(t *testing.T) {
	n := 10
	var nums []int
	for i := range n {
		nums = append(nums, i)
	}

	ite := slices.All(nums)
	for n := range ite {
		fmt.Println(n)
	}
}
