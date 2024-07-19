package snippet

import (
	"fmt"
	"slices"
	"strconv"
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

// Yield returns true if the iterator should continue
// with the next element in the sequence,
// false if it should stop.
func iter1(yield func(int, string) bool) {
	for i := range 3 {
		if !yield(i, strconv.Itoa(i)) {
			break
		}
	}
}

func TestIter_2(t *testing.T) {
	for k, v := range iter1 {
		fmt.Println("iter", k, v)
	}
}
