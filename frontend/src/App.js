import { useEffect, useState } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import { Card, CardContent } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Trophy, Zap, Users, TrendingUp } from "lucide-react";

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8001';
const API = `${API_BASE}/api`;

const HotOrNot = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [voteLoading, setVoteLoading] = useState(false);
  const [leaderboard, setLeaderboard] = useState([]);
  const [showLeaderboard, setShowLeaderboard] = useState(false);

  const fetchRandomPair = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/models/random`);
      setModels(response.data);
    } catch (error) {
      console.error('Error fetching random pair:', error);
      // If no models exist, seed them
      if (error.response?.status === 400) {
        await seedModels();
        fetchRandomPair();
      }
    } finally {
      setLoading(false);
    }
  };

  const seedModels = async () => {
    try {
      await axios.post(`${API}/models/seed`);
    } catch (error) {
      console.error('Error seeding models:', error);
    }
  };

  const fetchLeaderboard = async () => {
    try {
      const response = await axios.get(`${API}/models/leaderboard`);
      setLeaderboard(response.data);
    } catch (error) {
      console.error('Error fetching leaderboard:', error);
    }
  };

  const vote = async (winnerId, loserId) => {
    setVoteLoading(true);
    try {
      const response = await axios.post(`${API}/vote`, {
        winner_id: winnerId,
        loser_id: loserId
      });
      console.log(response.data.message);
      fetchRandomPair(); // Get new pair after voting
    } catch (error) {
      console.error('Error voting:', error);
    } finally {
      setVoteLoading(false);
    }
  };

  useEffect(() => {
    fetchRandomPair();
    fetchLeaderboard();
  }, []);

  if (showLeaderboard) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-white mb-2">🏆 LLM Leaderboard</h1>
            <p className="text-gray-300">The hottest AI models ranked by ELO rating</p>
          </div>

          <Button
            onClick={() => setShowLeaderboard(false)}
            className="mb-6 bg-purple-600 hover:bg-purple-700"
          >
            ← Back to Voting
          </Button>

          <div className="space-y-4">
            {leaderboard.map((model, index) => (
              <Card key={model.id} className="bg-white/10 border-white/20">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className="text-2xl font-bold text-white">#{index + 1}</div>
                      {index < 3 && <Trophy className="text-yellow-500 w-6 h-6" />}
                      <img
                        src={model.avatar_url}
                        alt={model.name}
                        className="w-16 h-16 rounded-full object-cover"
                      />
                      <div>
                        <h3 className="text-xl font-semibold text-white">{model.name}</h3>
                        <p className="text-gray-300">{model.provider}</p>
                        <p className="text-gray-400 text-sm">{model.parameters}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-white">{Math.round(model.rating)}</div>
                      <div className="text-sm text-gray-300">ELO Rating</div>
                      <div className="text-sm text-gray-400">{model.wins}W / {model.losses}L</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-2">🔥 LLM Hot or Not</h1>
          <p className="text-xl text-gray-300 mb-4">Which AI model is superior? You decide!</p>
          <div className="flex justify-center space-x-4">
            <Button
              onClick={() => setShowLeaderboard(true)}
              className="bg-yellow-600 hover:bg-yellow-700"
            >
              <Trophy className="w-4 h-4 mr-2" />
              Leaderboard
            </Button>
            <Button
              onClick={fetchRandomPair}
              className="bg-green-600 hover:bg-green-700"
              disabled={loading}
            >
              <Zap className="w-4 h-4 mr-2" />
              New Pair
            </Button>
          </div>
        </div>

        {loading ? (
          <div className="text-center text-white">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-white mx-auto mb-4"></div>
            <p>Loading amazing AI models...</p>
          </div>
        ) : models.length === 2 ? (
          <div className="grid md:grid-cols-2 gap-8">
            {models.map((model, index) => (
              <Card key={model.id} className="bg-white/10 backdrop-blur-lg border-white/20 hover:bg-white/15 transition-all duration-300">
                <CardContent className="p-8 text-center">
                  <img
                    src={model.avatar_url}
                    alt={model.name}
                    className="w-32 h-32 rounded-full mx-auto mb-4 object-cover shadow-lg"
                  />
                  <h2 className="text-2xl font-bold text-white mb-2">{model.name}</h2>
                  <Badge className="bg-blue-600 mb-3">{model.provider}</Badge>
                  <p className="text-gray-300 mb-2">{model.parameters}</p>
                  <p className="text-gray-400 text-sm mb-4">{model.description}</p>

                  <div className="flex justify-center space-x-4 mb-6 text-sm">
                    <div className="text-center">
                      <div className="text-white font-semibold">{Math.round(model.rating)}</div>
                      <div className="text-gray-400">ELO</div>
                    </div>
                    <div className="text-center">
                      <div className="text-white font-semibold">{model.wins}</div>
                      <div className="text-gray-400">Wins</div>
                    </div>
                    <div className="text-center">
                      <div className="text-white font-semibold">{model.votes}</div>
                      <div className="text-gray-400">Votes</div>
                    </div>
                  </div>

                  <Button
                    onClick={() => vote(model.id, models[1 - index].id)}
                    className="w-full bg-gradient-to-r from-pink-500 to-red-500 hover:from-pink-600 hover:to-red-600 text-white font-bold py-3 px-6 text-lg"
                    disabled={voteLoading}
                  >
                    {voteLoading ? (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    ) : (
                      <>🔥 This one is HOT! 🔥</>
                    )}
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <div className="text-center text-white">
            <p>No models available for comparison</p>
          </div>
        )}

        <div className="text-center mt-12 text-gray-400">
          <p className="text-sm">Powered by ELO rating system • Vote to influence the rankings</p>
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HotOrNot />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
