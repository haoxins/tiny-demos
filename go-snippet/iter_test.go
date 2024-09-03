package snippet

import (
	"fmt"
	"strconv"
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

type Result struct {
	Index int
}

func Scan(s *[]Result) {
	for i := range 10 {
		*s = append(*s, Result{Index: i})
	}
}

// func Iter() iter.Seq[Result] {
// }

func TestScanAndIter(t *testing.T) {
	var s []Result
	Scan(&s)

	fmt.Println(s)
}
