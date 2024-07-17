package snippet

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestArr(t *testing.T) {
	a := [...]int{1, 0, 0, 8, 6}
	assert.Equal(t, 5, len(a))
	assert.Equal(t, 5, cap(a))
	assert.Equal(t, 8, a[3])
}
