package snippet

import (
	"fmt"
	"testing"

	_ "github.com/stretchr/testify/assert"
)

func TestRange(t *testing.T) {
	for i := range 10 {
		fmt.Println(10 - i)
	}

	list := []string{"hello", "world"}
	for k := range list {
		fmt.Println(list[k])
	}
}

func TestRange2(t *testing.T) {
	s := []string{"hello", "world"}
	for i, x := range Backward(s) {
		fmt.Println(i, x)
	}
}

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
