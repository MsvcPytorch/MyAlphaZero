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
int MCTS_SIMULATIONS = 500;

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
const int batch_size = 512;
#include <cstdlib>  // 包含环境变量设置函数
class ModelEvaluator {
public:
	ModelEvaluator(const wchar_t* model_path)
		: env_(ORT_LOGGING_LEVEL_VERBOSE, "MCTS_Value_Network") {  // 设置日志级别为 VERBOSE
		try {
			// 创建 SessionOptions
			Ort::SessionOptions session_options;
			session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);  // 启用所有图优化

			// 创建 TensorRT Provider 选项
			OrtTensorRTProviderOptionsV2* tensorrt_options = nullptr;
			Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&tensorrt_options));

			// 设置 FP16 选项
			const char* trt_keys[] = { "trt_fp16_enable" };
			const char* trt_values[] = { "1" };
			Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(
				tensorrt_options, trt_keys, trt_values, 1));

			// 将 TensorRT Execution Provider 添加到会话选项中
			Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_TensorRT_V2(session_options, tensorrt_options));

			// 释放 TensorRT Provider 选项（在不需要时）
			Ort::GetApi().ReleaseTensorRTProviderOptions(tensorrt_options);

			// 创建会话
			std::cout << "Available Execution Providers before session creation:" << std::endl;
			for (const auto& provider : Ort::GetAvailableProviders()) {
				std::cout << "  " << provider << std::endl;
			}

			std::cout << "Loading model on GPU with TensorRT and FP16..." << std::endl;
			session_ = std::make_unique<Ort::Session>(env_, model_path, session_options);
			std::cout << "Model loaded successfully with TensorRT and FP16." << std::endl;

			// 获取输入节点名称
			size_t num_input_nodes = session_->GetInputCount();
			input_node_names_.reserve(num_input_nodes);
			input_node_name_ptrs_.reserve(num_input_nodes);
			Ort::AllocatorWithDefaultOptions allocator;
			for (size_t i = 0; i < num_input_nodes; ++i) {
				Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(i, allocator);
				input_node_names_.emplace_back(input_name.get());
				input_node_name_ptrs_.emplace_back(input_node_names_.back().c_str());
			}

			// 获取输出节点名称
			size_t num_output_nodes = session_->GetOutputCount();
			output_node_names_.reserve(num_output_nodes);
			output_node_name_ptrs_.reserve(num_output_nodes);
			for (size_t i = 0; i < num_output_nodes; ++i) {
				Ort::AllocatedStringPtr output_name = session_->GetOutputNameAllocated(i, allocator);
				output_node_names_.emplace_back(output_name.get());
				output_node_name_ptrs_.emplace_back(output_node_names_.back().c_str());
			}

			// 验证 TensorRT Execution Provider 是否被使用
			// std::cout << "Session Execution Providers:" << std::endl;
			// for (const auto& provider : session_->GetProviders()) {
			//     std::cout << "  " << provider << std::endl;
			// }
		}
		catch (const Ort::Exception& e) {
			std::cerr << "加载模型时出错: " << e.what() << std::endl;
			throw;
		}
	}

	~ModelEvaluator() {
		// 无需手动释放，std::string 会自动管理内存
	}

	std::pair<std::vector<double>, std::vector<std::vector<float>>> evaluate(
		const std::vector<std::vector<float>>& one_hot_boards     // size = 512 * 8*8*4
	) {
		try {
			const size_t board_size = SIZE * SIZE * 4;
			const size_t batch_size = one_hot_boards.size();
			// 检查输入大小
			if (batch_size != batch_size) {
				throw std::runtime_error("one_hot_boards size does not match batch_size (512).");
			}

			for (size_t i = 0; i < batch_size; ++i) {
				if (one_hot_boards[i].size() != board_size) {
					throw std::runtime_error("Each one_hot_board must have size 8*8*4.");
				}
			}

			// 扁平化输入数据
			std::vector<float> flattened_one_hot_boards;
			flattened_one_hot_boards.reserve(batch_size * board_size);
			for (const auto& board : one_hot_boards) {
				flattened_one_hot_boards.insert(flattened_one_hot_boards.end(), board.begin(), board.end());
			}

			// 定义输入形状
			std::vector<int64_t> board_input_shape = { static_cast<int64_t>(batch_size), 8, 8, 4 };

			// 创建输入张量
			Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
			std::vector<Ort::Value> input_tensors;

			// 创建 board 输入张量
			input_tensors.emplace_back(
				Ort::Value::CreateTensor<float>(
					memory_info,
					flattened_one_hot_boards.data(),
					flattened_one_hot_boards.size(),
					board_input_shape.data(),
					board_input_shape.size()
				)
			);
			// 运行推理
			auto output_tensors = session_->Run(
				Ort::RunOptions{ nullptr },
				input_node_name_ptrs_.data(),
				input_tensors.data(),
				input_tensors.size(),
				output_node_name_ptrs_.data(),
				output_node_name_ptrs_.size()
			);
			// 处理 Value 输出
			float* value_output_data = output_tensors[0].GetTensorMutableData<float>();
			std::vector<double> value_predictions(batch_size);
			for (size_t i = 0; i < batch_size; ++i) {
				value_predictions[i] = static_cast<double>(value_output_data[i]);
			}

			// 处理 Policy 输出
			float* policy_output_data = output_tensors[1].GetTensorMutableData<float>();
			size_t policy_size = 8 * 8 * 3;
			std::vector<std::vector<float>> policies(batch_size, std::vector<float>(policy_size));
			for (size_t i = 0; i < batch_size; ++i) {
				policies[i].assign(policy_output_data + i * policy_size, policy_output_data + (i + 1) * policy_size);
			}
			return { value_predictions, policies };
		}
		catch (const Ort::Exception& e) {
			std::cerr << "Error during model evaluation: " << e.what() << std::endl;
			throw;
		}
	}

private:
	Ort::Env env_;
	std::unique_ptr<Ort::Session> session_;
	std::vector<std::string> input_node_names_;
	std::vector<const char*> input_node_name_ptrs_;
	std::vector<std::string> output_node_names_;
	std::vector<const char*> output_node_name_ptrs_;
};
// 定义游戏状态结构体
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
pair<vector<double>, vector<vector<float> > > evaluateBoard(const vector<vector<vector<Player>>>& board, ModelEvaluator& evaluator, vector<Player> curPlayer) {
	// 将棋盘状态转换为一热编码
	vector<vector<float> > boards;
	for (int i = 0; i < batch_size; i++) {
		vector<float> one_hot_board;
		one_hot_board.reserve(SIZE * SIZE * 4);
		if (curPlayer[i] == RED) {
			for (int j = 0; j < SIZE; j++) {
				for (int p = 0; p < SIZE; p++) {
					int vl = -1;
					if (board[i][j][p] == BLOCK)vl = 0;
					else if (board[i][j][p] == EMPTY)vl = 1;
					else if (board[i][j][p] == curPlayer[i])vl = 2;
					else vl = 3;
					for (int z = 0; z < 4; z++)one_hot_board.push_back(z == vl);
				}
			}
		}
		else {
			for (int j = SIZE - 1; j >= 0; j--) {
				for (int p = 0; p < SIZE; p++) {
					int vl = -1;
					if (board[i][j][p] == BLOCK)vl = 0;
					else if (board[i][j][p] == EMPTY)vl = 1;
					else if (board[i][j][p] == curPlayer[i])vl = 2;
					else vl = 3;
					for (int z = 0; z < 4; z++)one_hot_board.push_back(z == vl);
				}
			}
		}
		boards.push_back(one_hot_board);
	}
	pair<vector<double>, vector<vector<float> > > ret = evaluator.evaluate(boards);
	for (int i = 0; i < batch_size; i++) {
		if (curPlayer[i] == BLUE) {
			for (int j = 0; j < SIZE / 2; j++) {
				for (int p = 0; p < SIZE; p++) {
					for (int q = 0; q < 3; q++)swap(ret.second[i][(j * SIZE + p) * 3 + q], ret.second[i][((SIZE - 1 - j) * SIZE + p) * 3 + q]);
				}
			}
		}
	}
	return ret;
}
mt19937 rd(time(0));
vector<float> generateDirichletNoice(double alpha, int len) {
	std::gamma_distribution<float> gamma(alpha, 1.0f);
	std::default_random_engine rng(std::time(nullptr));
	vector<float> ret; ret.resize(len);
	double sm = 0;
	for (int i = 0; i < len; i++)ret[i] = gamma(rng), sm += ret[i];
	if (len != 0 and sm <= 0.000001) {
		ret[0] += 0.0001;
		sm += 0.0001;
	}
	for (int i = 0; i < len; i++)ret[i] = ret[i] / sm;
	return ret;
}
int disRed[batch_size][SIZE][SIZE];
int disBlue[batch_size][SIZE][SIZE];
double T, dt;
// 扩展节点,返回评估值
vector<double> expand(vector<MCTSNode*> nodes, ModelEvaluator& evaluator, bool isFirst) {
	vector<double> ret; ret.resize(batch_size);
	vector<vector<vector<Player>>> board; vector<Player> curPlayer;
	for (int i = 0; i < batch_size; i++) {
		board.push_back(nodes[i]->state.board);
		curPlayer.push_back(nodes[i]->state.currentPlayer);
	}
	pair<vector<double>, vector <vector< float> > > eval = evaluateBoard(board, evaluator, curPlayer);
	for (int i = 0; i < batch_size; i++) {
		ret[i] = -eval.first[i];
		int res = nodes[i]->state.judgeWin();
		if (res != 0) {
			if (nodes[i]->state.currentPlayer == RED)ret[i] = -res;
			else ret[i] = res;
			continue;
		}
		vector<mov> moves = nodes[i]->state.getavailableMoves();
		if (moves.empty()) {
			ret[i] = 0;
			continue;
		}
		if (isFirst) {
			vector<float> DNoice = generateDirichletNoice(0.03, moves.size());
			double sm = 0;
			for (int j = 0; j < eval.second[i].size(); j++)sm += pow(eval.second[i][j], 1.00 / T);
			for (int j = 0; j < eval.second[i].size(); j++)eval.second[i][j] = 1.00 * pow(eval.second[i][j], 1.00 / T) / sm;//温度控制	
			int ptr = 0;
			for (auto v : moves) {
				int tp = v.second.second - v.first.second + 1;
				eval.second[i][(v.first.first * SIZE + v.first.second) * 3 + tp] = eval.second[i][(v.first.first * SIZE + v.first.second) * 3 + tp] * 0.75 + DNoice[ptr++] * 0.25;
			}
		}
		for (auto v : moves) {
			int tp = v.second.second - v.first.second + 1;
			nodes[i]->lazySons.push_back(make_pair(eval.second[i][(v.first.first * SIZE + v.first.second) * 3 + tp], v));
		}
		sort(nodes[i]->lazySons.begin(), nodes[i]->lazySons.end());
		double score = eval.first[i];
	}
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

float policyTarget[batch_size][SIZE][SIZE][3];
// AI 使用 MCTS-UCT 选择最佳移动，并返回胜率
vector<mov> getBestMoveMCTS(const vector<GameState>& rootState, vector<double>& winRate, ModelEvaluator& evaluator) {
	// 创建根节点
	winRate.resize(batch_size);
	vector<MCTSNode*> root; root.resize(batch_size);
	for (int i = 0; i < batch_size; i++) {
		root[i] = new MCTSNode(rootState[i]);
	}
	// 进行一定次数的模拟
	vector<MCTSNode*> nodes; nodes.resize(batch_size);
	for (int i = 0; i < MCTS_SIMULATIONS; i++) {
		// Selection
		for (int j = 0; j < batch_size; j++) {
			nodes[j] = root[j];
			while (!nodes[j]->isLeaf()) {
				nodes[j] = selectChild(nodes[j]);
				if (nodes[j] == nullptr) break;
			}
		}
		// Simulation
		vector<double> result = expand(nodes, evaluator, i == 0);
		// Backpropagation
		for (int j = 0; j < batch_size; j++)backpropagate(nodes[j], result[j]);
	}
	// 计算胜率
	vector<mov> ret; ret.resize(batch_size);
	for (int i = 0; i < batch_size; i++) {
		if (root[i]->visits > 0) {
			winRate[i] = 1.00 - (root[i]->visits + root[i]->wins) / root[i]->visits / 2;
		}
		else {
			winRate[i] = 0.0;
		}
		for (int j = 0; j < SIZE; j++) {
			for (int p = 0; p < SIZE; p++) {
				for (int q = 0; q < 3; q++)policyTarget[i][j][p][q] = 0;
			}
		}
		for (auto v : root[i]->children) {
			int tp = v->move.second.second - v->move.first.second + 1;
			policyTarget[i][v->move.first.first][v->move.first.second][tp] = 1.00 * v->visits / (root[i]->visits - 1);
		}
		// 选择访问次数最多的子节点
		ret[i] = bestMove(root[i]);
		// 清理内存
		delete root[i];
	}
	return ret;
}
// 主要游戏函数
bool isFinish[batch_size];
int main() {
	ios::sync_with_stdio(0); cin.tie(0); cout.tie(0);
	srand(static_cast<unsigned int>(time(0))); // 初始化随机种子

	// 初始化棋盘状态
	GameState rootState;

	// 初始化模型评估器
	const wchar_t* model_path = L"model_fp16.onnx"; // 请确保路径正确
	ModelEvaluator evaluator(model_path);
	int epochNum = 20;
	freopen("data.csv", "w", stdout);
	cout << "board_000,board_001,board_002,board_003,board_010,board_011,board_012,board_013,board_020,board_021,board_022,board_023,board_030,board_031,board_032,board_033,board_040,board_041,board_042,board_043,board_050,board_051,board_052,board_053,board_060,board_061,board_062,board_063,board_070,board_071,board_072,board_073,board_100,board_101,board_102,board_103,board_110,board_111,board_112,board_113,board_120,board_121,board_122,board_123,board_130,board_131,board_132,board_133,board_140,board_141,board_142,board_143,board_150,board_151,board_152,board_153,board_160,board_161,board_162,board_163,board_170,board_171,board_172,board_173,board_200,board_201,board_202,board_203,board_210,board_211,board_212,board_213,board_220,board_221,board_222,board_223,board_230,board_231,board_232,board_233,board_240,board_241,board_242,board_243,board_250,board_251,board_252,board_253,board_260,board_261,board_262,board_263,board_270,board_271,board_272,board_273,board_300,board_301,board_302,board_303,board_310,board_311,board_312,board_313,board_320,board_321,board_322,board_323,board_330,board_331,board_332,board_333,board_340,board_341,board_342,board_343,board_350,board_351,board_352,board_353,board_360,board_361,board_362,board_363,board_370,board_371,board_372,board_373,board_400,board_401,board_402,board_403,board_410,board_411,board_412,board_413,board_420,board_421,board_422,board_423,board_430,board_431,board_432,board_433,board_440,board_441,board_442,board_443,board_450,board_451,board_452,board_453,board_460,board_461,board_462,board_463,board_470,board_471,board_472,board_473,board_500,board_501,board_502,board_503,board_510,board_511,board_512,board_513,board_520,board_521,board_522,board_523,board_530,board_531,board_532,board_533,board_540,board_541,board_542,board_543,board_550,board_551,board_552,board_553,board_560,board_561,board_562,board_563,board_570,board_571,board_572,board_573,board_600,board_601,board_602,board_603,board_610,board_611,board_612,board_613,board_620,board_621,board_622,board_623,board_630,board_631,board_632,board_633,board_640,board_641,board_642,board_643,board_650,board_651,board_652,board_653,board_660,board_661,board_662,board_663,board_670,board_671,board_672,board_673,board_700,board_701,board_702,board_703,board_710,board_711,board_712,board_713,board_720,board_721,board_722,board_723,board_730,board_731,board_732,board_733,board_740,board_741,board_742,board_743,board_750,board_751,board_752,board_753,board_760,board_761,board_762,board_763,board_770,board_771,board_772,board_773,policy_000,policy_001,policy_002,policy_010,policy_011,policy_012,policy_020,policy_021,policy_022,policy_030,policy_031,policy_032,policy_040,policy_041,policy_042,policy_050,policy_051,policy_052,policy_060,policy_061,policy_062,policy_070,policy_071,policy_072,policy_100,policy_101,policy_102,policy_110,policy_111,policy_112,policy_120,policy_121,policy_122,policy_130,policy_131,policy_132,policy_140,policy_141,policy_142,policy_150,policy_151,policy_152,policy_160,policy_161,policy_162,policy_170,policy_171,policy_172,policy_200,policy_201,policy_202,policy_210,policy_211,policy_212,policy_220,policy_221,policy_222,policy_230,policy_231,policy_232,policy_240,policy_241,policy_242,policy_250,policy_251,policy_252,policy_260,policy_261,policy_262,policy_270,policy_271,policy_272,policy_300,policy_301,policy_302,policy_310,policy_311,policy_312,policy_320,policy_321,policy_322,policy_330,policy_331,policy_332,policy_340,policy_341,policy_342,policy_350,policy_351,policy_352,policy_360,policy_361,policy_362,policy_370,policy_371,policy_372,policy_400,policy_401,policy_402,policy_410,policy_411,policy_412,policy_420,policy_421,policy_422,policy_430,policy_431,policy_432,policy_440,policy_441,policy_442,policy_450,policy_451,policy_452,policy_460,policy_461,policy_462,policy_470,policy_471,policy_472,policy_500,policy_501,policy_502,policy_510,policy_511,policy_512,policy_520,policy_521,policy_522,policy_530,policy_531,policy_532,policy_540,policy_541,policy_542,policy_550,policy_551,policy_552,policy_560,policy_561,policy_562,policy_570,policy_571,policy_572,policy_600,policy_601,policy_602,policy_610,policy_611,policy_612,policy_620,policy_621,policy_622,policy_630,policy_631,policy_632,policy_640,policy_641,policy_642,policy_650,policy_651,policy_652,policy_660,policy_661,policy_662,policy_670,policy_671,policy_672,policy_700,policy_701,policy_702,policy_710,policy_711,policy_712,policy_720,policy_721,policy_722,policy_730,policy_731,policy_732,policy_740,policy_741,policy_742,policy_750,policy_751,policy_752,policy_760,policy_761,policy_762,policy_770,policy_771,policy_772,label" << endl;
	for (int _ = 0; _ < epochNum; _++) {
		vector<GameState> rootState; rootState.resize(batch_size);
		for (int i = 0; i < batch_size; i++) {
			isFinish[i] = 0;
			int C = rd() % 4;
			while (C--) {
				int x, y;
				do {
					x = rd() % (SIZE - 4) + 2;
					y = rd() % SIZE;
				} while (rootState[i].board[x][y] != EMPTY);
				rootState[i].board[x][y] = BLOCK;
			}
			C = 2 * (rd() % 4);
			while (C--) {
				vector<mov> availMoves = rootState[i].getavailableMoves();
				if (availMoves.empty())break;
				mov o = availMoves[rd() % (int)availMoves.size()];
				rootState[i].executeMove(o.first.first, o.first.second, o.second.first, o.second.second);
			}
		}
		cerr << "Starting Epoch " << _ << endl;
		vector<vector<vector<double> > > sta; sta.resize(batch_size);
		T = 1.80; dt = 0.98;
		for (int __ = 1; ; __++) {
			cerr << "Starting Epoch " << _ << " Turn " << __ << endl;
			vector<double> winRate;
			vector<vector<double> > e; e.resize(batch_size);
			for (int i = 0; i < batch_size; i++) {
				vector<double> one_hot_board;
				one_hot_board.reserve(SIZE * SIZE * 4);
				if (rootState[i].currentPlayer == RED) {
					for (int j = 0; j < SIZE; j++) {
						for (int p = 0; p < SIZE; p++) {
							int vl = -1;
							if (rootState[i].board[j][p] == BLOCK)vl = 0;
							else if (rootState[i].board[j][p] == EMPTY)vl = 1;
							else if (rootState[i].board[j][p] == rootState[i].currentPlayer)vl = 2;
							else vl = 3;
							for (int z = 0; z < 4; z++)one_hot_board.push_back(z == vl);
						}
					}
				}
				else {
					for (int j = SIZE - 1; j >= 0; j--) {
						for (int p = 0; p < SIZE; p++) {
							int vl = -1;
							if (rootState[i].board[j][p] == BLOCK)vl = 0;
							else if (rootState[i].board[j][p] == EMPTY)vl = 1;
							else if (rootState[i].board[j][p] == rootState[i].currentPlayer)vl = 2;
							else vl = 3;
							for (int z = 0; z < 4; z++)one_hot_board.push_back(z == vl);
						}
					}
				}
				e[i] = one_hot_board;
			}
			vector<mov> aiMove = getBestMoveMCTS(rootState, winRate, evaluator); T *= dt; T = max(T, 1.10);
			bool isShowed = 0;
			for (int i = 0; i < batch_size; i++) {
				if (rootState[i].currentPlayer == RED) {
					for (int j = 0; j < SIZE; j++) {
						for (int p = 0; p < SIZE; p++) {
							for (int q = 0; q < 3; q++)e[i].push_back(policyTarget[i][j][p][q]);
						}
					}
				}
				else {
					for (int j = SIZE - 1; j >= 0; j--) {
						for (int p = 0; p < SIZE; p++) {
							for (int q = 0; q < 3; q++)e[i].push_back(policyTarget[i][j][p][q]);
						}
					}
				}
				if (!isFinish[i]) {
					if (!isShowed) {
						isShowed = 1;
						displayBoard(rootState[i].board);
						cerr << (rootState[i].currentPlayer == RED ? "Red" : "Blue") << "WinRate : " << winRate[i] << endl;
					}
					sta[i].push_back(e[i]);
					rootState[i].executeMove(aiMove[i].first.first, aiMove[i].first.second, aiMove[i].second.first, aiMove[i].second.second);
					if (rootState[i].judgeWin() != 0)isFinish[i] = 1;
				}
			}
			cerr << "Finished Epoch " << _ << " Turn " << __ << endl;
			bool allOver = 1;
			for (int i = 0; i < batch_size; i++) {
				if (!isFinish[i])allOver = 0;
			}
			if (allOver)break;
			if (__ == 150)break;
		}
		for (int i = 0; i < batch_size; i++) {
			double win;
			if (isFinish[i])win = rootState[i].judgeWin();
			else win = 0;
			for (int j = 0; j < sta[i].size(); j++) {
				for (int p = 0; p < sta[i][j].size(); p++)cout << sta[i][j][p] << ",";
				cout << (j % 2 == 0 ? win : -win) << endl;
			}
		}
	}
	return 0;
}
