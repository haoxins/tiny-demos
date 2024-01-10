package snippet

import (
	"fmt"
	"testing"

	_ "github.com/stretchr/testify/assert"
)

func Backward[E any](s []E) func(func(int, E) bool) {
	return func(yield func(int, E) bool) {
		for i := len(s) - 1; i >= 0; i-- {
			if !yield(i, s[i]) {
				return
			}
		}
		return
	}
}

func TestRange(t *testing.T) {
	for i := range 10 {
		fmt.Print(10-i, " ")
	}
}

func TestRange2(t *testing.T) {
	s := []string{"hello", "world"}
	for i, x := range Backward(s) {
		fmt.Println(i, x)
	}
}
