package snippet

import (
	"fmt"
	"slices"
	"strconv"
	"sync"
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

func TestIter_3(t *testing.T) {
	var m sync.Map

	m.Store("alice", 11)
	m.Store("bob", 12)
	m.Store("cindy", 13)

	// Go 1.22
	m.Range(func(key, val any) bool {
		fmt.Println(key, val)
		return true
	})

	// Go 1.23+
	for key, val := range m.Range {
		fmt.Println(key, val)
	}
}
