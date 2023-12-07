package snippet

import (
	"encoding/json"
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/stat"
)

func TestJSON(t *testing.T) {
	s1 := `{"": []}`
	a1 := Account{}
	json.Unmarshal([]byte(s1), &a1)
	assert.Equal(t, []string([]string(nil)), a1.Badges)
	assert.Equal(t, 0, len(a1.Badges))

	s2 := `{"badges": null}`
	a2 := Account{}
	json.Unmarshal([]byte(s2), &a2)
	assert.Equal(t, []string([]string(nil)), a2.Badges)
	assert.Equal(t, 0, len(a2.Badges))

	a1 = Account{
		Badges: nil,
	}
	b1, _ := json.Marshal(a1)
	assert.Equal(t, string(b1), `{"badges":null}`)

	a2 = Account{
		Badges: []string{},
	}
	b2, _ := json.Marshal(a2)
	assert.Equal(t, string(b2), `{"badges":[]}`)

	// NaN
	v := stat.Variance([]float64{0.0}, nil)
	assert.Equal(t, true, math.IsNaN(v))
	v = stat.Variance([]float64{1.1}, nil)
	assert.Equal(t, true, math.IsNaN(v))
	fmt.Println("variance:", v)
	_, e := json.Marshal(Transaction{
		Amount: v,
	})
	assert.Equal(t, "json: unsupported value: NaN", e.Error())
}
