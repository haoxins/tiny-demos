package snippet

import (
	"fmt"
	"slices"
	"testing"
)

func TestIter_1(t *testing.T) {
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
func iter1(yield func(int, int) bool) {
	for i := range 3 {
		if !yield(i, i+1) {
			return
		}
	}
}

func TestIter_2(t *testing.T) {
	for k, v := range iter1 {
		fmt.Println("iter", k, v)
	}
}
