package snippet

type Account struct {
	Badges  []string `json:"badges"`
	Balance *Balance `json:"balance"`
}

type Transaction struct {
	Amount float64 `json:"amount"`
}

type Balance struct {
	Amount float64 `json:"amount"`
}
