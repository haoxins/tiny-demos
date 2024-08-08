package snippet

import (
	"fmt"
	"testing"
)

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
