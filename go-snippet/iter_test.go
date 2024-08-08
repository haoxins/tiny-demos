package snippet

import (
	"fmt"
	"strconv"
	"sync"
	"testing"
)

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

func TestIter1(t *testing.T) {
	for k, v := range iter1 {
		fmt.Println("iter", k, v)
	}
}

func TestIter2(t *testing.T) {
	var m sync.Map

	m.Store("alice", 11)
	m.Store("bob", 12)
	m.Store("cindy", 13)

	for key, val := range m.Range {
		fmt.Println(key, val)
	}
}
