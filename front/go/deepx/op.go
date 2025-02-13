package deepx

// 添加操作类型枚举
type OpType int

const (
	OpAdd OpType = iota
	OpSub
	OpMul
	OpDiv
	OpMatmul
)

// 辅助函数转换操作类型
func opTypeToString(op OpType) string {
	switch op {
	case OpAdd:
		return "+"
	case OpSub:
		return "-"
	case OpMul:
		return "✖"
	case OpDiv:
		return "÷"
	case OpMatmul:
		return "MatMul"
	default:
		return "Unknown"
	}
}
