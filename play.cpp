// mcts_red_blue_chess.cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <limits>
#include <unordered_set>
#include <onnxruntime_cxx_api.h>
#include <random>

using namespace std;
enum Player { BLOCK, EMPTY, RED, BLUE };
// 定义 AI 的模拟次数
int MCTS_SIMULATIONS = 200000;

// 将棋子类型转换为字符
string pieceToChar(Player x) {
	if (x == BLOCK)return "#";
	if (x == EMPTY)return ".";
	if (x == RED)return "R";
	if (x == BLUE)return "B";
}
const int SIZE = 8;
// 显示棋盘
void displayBoard(const vector<vector<Player>>& board) {
	cerr << "  ";
	for (int c = 0; c < SIZE; c++) cerr << c << " ";
	cerr << endl;
	for (int r = 0; r < SIZE; r++) {
		cerr << r << " ";
		for (int c = 0; c < SIZE; c++) {
			cerr << pieceToChar(board[r][c]) << " ";
		}
		cerr << endl;
	}
}

// 检查坐标是否合法
bool isValid(int r, int c) {
	return r >= 0 && r < SIZE && c >= 0 && c < SIZE;
}

#include <memory>
#include <utility>
#include <stdexcept>
#include <cassert>
#include <cstdlib>  // 包含环境变量设置函数
class ModelEvaluator {
public:
	// 构造函数：加载模型并初始化会话
	ModelEvaluator(const wchar_t* model_path)
		: env_(ORT_LOGGING_LEVEL_WARNING, "MCTS_Value_Network"),
		session_(env_, model_path, Ort::SessionOptions{ nullptr }) {
		// 获取输入和输出节点名称
		size_t num_input_nodes = session_.GetInputCount();
		size_t num_output_nodes = session_.GetOutputCount();

		input_node_names_.resize(num_input_nodes);
		output_node_names_.resize(num_output_nodes);

		// 预留空间以避免多次分配
		input_node_names_alloc_.reserve(num_input_nodes);
		output_node_names_alloc_.reserve(num_output_nodes);

		Ort::AllocatorWithDefaultOptions allocator;
		for (size_t i = 0; i < num_input_nodes; ++i) {
			// 使用 GetInputNameAllocated 获取输入名，并存储以管理内存
			Ort::AllocatedStringPtr input_name = session_.GetInputNameAllocated(i, allocator);
			input_node_names_alloc_.emplace_back(std::move(input_name));
			input_node_names_[i] = input_node_names_alloc_[i].get();
		}

		for (size_t i = 0; i < num_output_nodes; ++i) {
			// 使用 GetOutputNameAllocated 获取输出名，并存储以管理内存
			Ort::AllocatedStringPtr output_name = session_.GetOutputNameAllocated(i, allocator);
			output_node_names_alloc_.emplace_back(std::move(output_name));
			output_node_names_[i] = output_node_names_alloc_[i].get();
		}

		std::wcout << L"ModelEvaluator initialized successfully." << std::endl;
	}

	// 移除析构函数中的手动内存释放
	~ModelEvaluator() {
		// Ort::AllocatedStringPtr 会自动管理内存，无需手动释放
	}

	// 推理函数：接受单个样本的输入，返回输出
	std::pair<double, std::vector<float>> evaluate(const std::vector<float>& one_hot_board) {
		try {
			// 输入验证
			const size_t expected_board_size = 8 * 8 * 4;

			if (one_hot_board.size() != expected_board_size) {
				throw std::runtime_error("one_hot_board size mismatch. Expected size: 252.");
			}

			//std::wcout << L"Input validation passed." << std::endl;

			// 扁平化输入数据（已是扁平化）
			std::vector<float> flattened_one_hot_board = one_hot_board;

			// 定义输入形状
			std::vector<int64_t> board_input_shape = { 1, 8, 8, 4 };    // 批量大小为1

			// 创建输入张量
			Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
			std::vector<Ort::Value> input_tensors;

			// 创建 board 输入张量
			Ort::Value board_tensor = Ort::Value::CreateTensor<float>(
				memory_info,
				const_cast<float*>(flattened_one_hot_board.data()),
				flattened_one_hot_board.size(),
				board_input_shape.data(),
				board_input_shape.size()
			);
			input_tensors.emplace_back(std::move(board_tensor));

			//std::wcout << L"Created input tensors." << std::endl;

			// 进行推理
			//std::wcout << L"Running inference..." << std::endl;
			auto output_tensors = session_.Run(
				Ort::RunOptions{ nullptr },
				input_node_names_.data(),
				input_tensors.data(),
				input_tensors.size(),
				output_node_names_.data(),
				output_node_names_.size()
			);
			//std::wcout << L"Inference completed." << std::endl;

			// 获取 Value 输出
			float* value_output_data = output_tensors[0].GetTensorMutableData<float>();
			double value_prediction = static_cast<double>(value_output_data[0]);  // 模型的 Value 输出是一个标量

			// 获取 Policy 输出
			float* policy_output_data = output_tensors[1].GetTensorMutableData<float>();
			size_t policy_size = 8 * 8 * 3;
			std::vector<float> policy(policy_output_data, policy_output_data + policy_size);

			return { value_prediction, policy };
		}
		catch (const Ort::Exception& e) {
			std::cerr << "Error during inference: " << e.what() << std::endl;
			throw;
		}
		catch (const std::exception& e) {
			std::cerr << "Standard exception during inference: " << e.what() << std::endl;
			throw;
		}
	}

private:
	Ort::Env env_;
	Ort::Session session_;
	std::vector<const char*> input_node_names_;
	std::vector<const char*> output_node_names_;

	// 添加用于存储已分配字符串的成员变量，确保生命周期
	std::vector<Ort::AllocatedStringPtr> input_node_names_alloc_;
	std::vector<Ort::AllocatedStringPtr> output_node_names_alloc_;
};
#define mov pair<pair<int,int>,pair<int,int> >
struct GameState {
	vector<vector<Player> > board;
	Player currentPlayer;
	GameState() : board(SIZE, vector<Player>(SIZE)), currentPlayer(RED) {
		for (int i = 0; i < SIZE; i++) {
			for (int j = 0; j < SIZE; j++)board[i][j] = EMPTY;
		}
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < SIZE; j++) {
				board[i][j] = RED; board[SIZE - i - 1][j] = BLUE;
			}
		}
		return;
	}

	// 深拷贝构造函数
	GameState(const GameState& other) {
		board = other.board;
		currentPlayer = other.currentPlayer;
	}
	// 执行移动
	void executeMove(int x, int y, int nx, int ny) {
		board[nx][ny] = board[x][y]; board[x][y] = EMPTY;
		currentPlayer = (currentPlayer == RED) ? BLUE : RED;
		return;
	}
	int judgeWin() {
		int k1 = 0, k2 = 0;
		for (int i = 0; i < SIZE; i++) {
			if (board[SIZE - 1][i] == RED)k1 = 1;
			if (board[0][i] == BLUE)k2 = 1;
		}
		if (k1)return 1;
		if (k2)return -1;
		return 0;
	}

	// 检查游戏是否结束
	vector<mov> getavailableMoves() {
		vector<mov> ret;
		for (int i = 0; i < SIZE; i++) {
			for (int j = 0; j < SIZE; j++) {
				if (board[i][j] != currentPlayer)continue;
				for (int p = -1; p <= 1; p++) {
					int nx = i + (currentPlayer == RED ? 1 : -1);
					int ny = j + p;
					if (!isValid(nx, ny))continue;
					if (board[nx][ny] == BLOCK)continue;
					if (board[nx][ny] == currentPlayer)continue;
					if (p == 0 and board[nx][ny] != EMPTY)continue;
					ret.push_back({ {i,j},{nx,ny} });
				}
			}
		}
		return ret;
	}
};

// 定义 MCTS 节点
struct MCTSNode {
	GameState state;
	MCTSNode* parent;
	vector<MCTSNode*> children;
	mov move; // Move that led to this node
	int visits;
	double wins; // Wins for AI
	double prior;
	vector<pair<float, mov > > lazySons;
	MCTSNode(const GameState& state, MCTSNode* parent = nullptr, const mov& move = { { -1,-1 },{-1,-1} }, double prior = 1.0)
		: state(state), parent(parent), move(move), visits(0), wins(0.0), prior(prior) {}

	~MCTSNode() {
		for (auto child : children) {
			delete child;
		}
	}

	// 判断是否为叶子节点
	bool isLeaf() const {
		return children.empty() and lazySons.empty();
	}
};
// 模型现在返回的是当前方胜率
pair<double, vector<float> > evaluateBoard(const vector<vector<Player>>& board, ModelEvaluator& evaluator, Player curPlayer) {
	// 将棋盘状态转换为一热编码
	vector<float> one_hot_board;
	one_hot_board.reserve(SIZE * SIZE * 4);
	if (curPlayer == RED) {
		for (int j = 0; j < SIZE; j++) {
			for (int p = 0; p < SIZE; p++) {
				int vl = -1;
				if (board[j][p] == BLOCK)vl = 0;
				else if (board[j][p] == EMPTY)vl = 1;
				else if (board[j][p] == curPlayer)vl = 2;
				else vl = 3;
				for (int z = 0; z < 4; z++)one_hot_board.push_back(z == vl);
			}
		}
	}
	else {
		for (int j = SIZE - 1; j >= 0; j--) {
			for (int p = 0; p < SIZE; p++) {
				int vl = -1;
				if (board[j][p] == BLOCK)vl = 0;
				else if (board[j][p] == EMPTY)vl = 1;
				else if (board[j][p] == curPlayer)vl = 2;
				else vl = 3;
				for (int z = 0; z < 4; z++)one_hot_board.push_back(z == vl);
			}
		}
	}
	pair<double, vector<float> > ret = evaluator.evaluate(one_hot_board);
	if (curPlayer == BLUE) {
		for (int j = 0; j < SIZE / 2; j++) {
			for (int p = 0; p < SIZE; p++) {
				for (int q = 0; q < 3; q++)swap(ret.second[(j * SIZE + p) * 3 + q], ret.second[((SIZE - 1 - j) * SIZE + p) * 3 + q]);
			}
		}
	}
	return ret;
}
mt19937 rd(time(0));
double expand(MCTSNode* node, ModelEvaluator& evaluator, bool isFirst) {
	pair<double, vector <float> > eval = evaluateBoard(node->state.board, evaluator, node->state.currentPlayer);
	double ret;
	ret = -eval.first;
	int res = node->state.judgeWin();
	if (res != 0) {
		if (node->state.currentPlayer == RED)ret = -res;
		else ret = res;
		return ret;
	}
	vector<mov> moves = node->state.getavailableMoves();
	if (moves.empty()) {
		ret = 0.00; return ret;
	}
	for (auto v : moves) {
		int tp = v.second.second - v.first.second + 1;
		node->lazySons.push_back(make_pair(eval.second[(v.first.first * SIZE + v.first.second) * 3 + tp], v));
	}
	sort(node->lazySons.begin(), node->lazySons.end());
	return ret;
}
const double C_PUCT = 1.0;
// 选择最佳子节点（基于 UCT）
MCTSNode* selectChild(MCTSNode* node) {
	double bestPUCT = -1.0;
	MCTSNode* bestChild = nullptr;
	for (auto child : node->children) {
		double exploitation = child->wins / (1 + child->visits);
		double exploration = C_PUCT * child->prior * sqrt(node->visits) / (1 + child->visits);
		double puctValue = exploitation + exploration;
		if (puctValue > bestPUCT) {
			bestPUCT = puctValue;
			bestChild = child;
		}
	}
	if (!node->lazySons.empty()) {
		double puctValue = C_PUCT * node->lazySons.back().first * sqrt(node->visits);
		if (puctValue > bestPUCT) {
			bestPUCT = puctValue;
			mov v = node->lazySons.back().second;
			GameState newState = node->state;
			newState.executeMove(v.first.first, v.first.second, v.second.first, v.second.second);
			MCTSNode* child = new MCTSNode(newState, node, v, node->lazySons.back().first);
			node->children.push_back(child);
			node->lazySons.pop_back();
			bestChild = child;
		}
	}
	return bestChild;
}
// 反向传播结果
void backpropagate(MCTSNode* node, double result) {
	while (node != nullptr) {
		node->visits++;
		node->wins += result;
		node = node->parent;
		result = -result;
	}
}

// 选择最优移动（选择访问次数最多的子节点）
mov bestMove(MCTSNode* root) {
	double maxVisits = -1.0;
	MCTSNode* bestChild = nullptr;
	for (auto child : root->children) {
		if (child->visits > maxVisits) {
			maxVisits = child->visits;
			bestChild = child;
		}
	}
	if (bestChild == nullptr) {
		return { {-1,-1},{-1,-1} };
	}
	return bestChild->move;
}
// AI 使用 MCTS-UCT 选择最佳移动，并返回胜率
mov getBestMoveMCTS(const GameState& rootState, double& winRate, ModelEvaluator& evaluator) {
	// 创建根节点
	MCTSNode* root = new MCTSNode(rootState);
	// 进行一定次数的模拟
	MCTSNode* node;
	for (int i = 0; i < MCTS_SIMULATIONS; i++) {
		// Selection
		node = root;
		while (!node->isLeaf()) {
			node = selectChild(node);
			if (node == nullptr) break;
		}
		// Simulation
		double result = expand(node, evaluator, i == 0);
		// Backpropagation
		backpropagate(node, result);
		if (i % 10000 == 0) {
			for (auto v : root->children) {
				cerr << "Move " << v->move.first.first << " " << v->move.first.second << " " << v->move.second.first << " " << v->move.second.second << " prior " << v->prior << " visits " << v->visits << " wr " << (v->visits + v->wins) / v->visits / 2 << endl;
			}
			cout << endl;
		}
	}
	// 计算胜率
	mov ret;
	if (root->visits > 0) {
		winRate = 1.00 - (root->visits + root->wins) / root->visits / 2;
	}
	else {
		winRate = 0.0;
	}
	// 选择访问次数最多的子节点
	ret = bestMove(root);
	// 清理内存
	delete root;
	return ret;
}
// 主要游戏函数
int main() {
	ios::sync_with_stdio(0); cin.tie(0); cout.tie(0);
	srand(static_cast<unsigned int>(time(0))); // 初始化随机种子

	// 初始化棋盘状态

	// 初始化模型评估器
	const wchar_t* model_path = L"model_fp16.onnx"; // 请确保路径正确
	ModelEvaluator evaluator(model_path);
	GameState rootState;
	int C; cin >> C;
	while (C--) {
		int x, y; cin >> x >> y;
		rootState.board[x][y] = BLOCK;
	}
	int tp;
	cout << "0红1蓝" << endl;
	cin >> tp;
	while (1) {
		displayBoard(rootState.board);
		if (tp) {
			double wr;
			mov res = getBestMoveMCTS(rootState, wr, evaluator);
			cout << "MOVE " << res.first.first << " " << res.first.second << " " << res.second.first << " " << res.second.second << endl;
			cout << "WR " << wr << endl;
			rootState.executeMove(res.first.first, res.first.second, res.second.first, res.second.second);
		}
		else {
			cout << "轮到你了" << endl;
			int x, y, z, w; cin >> x >> y >> z >> w;
			while (!isValid(x, y) or !isValid(z, w))cin >> x >> y >> z >> w;
			rootState.executeMove(x, y, z, w);
		}
		tp ^= 1;
	}
	return 0;
}

