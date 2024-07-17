package snippet

import (
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRepeat(t *testing.T) {
	a := [...]int{1, 2, 3}
	b := slices.Repeat(a[:], 1)
	c := a[:]

	b[1] = 100
	c[1] = 200

	assert.Equal(t, [3]int{1, 200, 3}, a)
	assert.Equal(t, []int{1, 100, 3}, b)
	assert.Equal(t, []int{1, 200, 3}, c)
}
