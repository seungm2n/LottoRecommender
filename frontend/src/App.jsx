import React, { useState, useEffect } from "react";
import axios from "axios";
import "./reset.css";
import "./common.css";

const numbersGrid = Array.from({ length: 7 }, (_, row) =>
    Array.from({ length: 7 }, (_, col) => row * 7 + col + 1)
);

function getBallColorClass(num) {
    if (num >= 1 && num <= 10) return "yellow";
    if (num >= 11 && num <= 20) return "blue";
    if (num >= 21 && num <= 30) return "red";
    if (num >= 31 && num <= 40) return "black";
    if (num >= 41 && num <= 45) return "green";
    return "";
}

function App() {
    const [count, setCount] = useState(1);
    const [sets, setSets] = useState([]);
    const [loading, setLoading] = useState(false);
    const [excluded, setExcluded] = useState([]);
    const [included, setIncluded] = useState([]);
    const [recommendType, setRecommendType] = useState("normal");
    const [latestRound, setLatestRound] = useState(null);
    const [startRound, setStartRound] = useState(1);
    const [endRound, setEndRound] = useState(1);
    const [showModal, setShowModal] = useState(false);
    const [elapsed, setElapsed] = useState(0);

    useEffect(() => {
        async function fetchLatestRound() {
            try {
                const res = await axios.get("http://localhost:8000/lotto/latest-round");
                if (res.data.latest_round) {
                    setLatestRound(res.data.latest_round);
                    setStartRound(res.data.latest_round-100);
                    setEndRound(res.data.latest_round);
                }
            } catch (error) {
                console.error("최신 회차 정보를 가져오는 중 에러:", error);
            }
        }
        fetchLatestRound();
    }, []);

    useEffect(() => {
        let timer;
        if (loading) {
            setElapsed(0);
            timer = setInterval(() => setElapsed((prev) => prev + 1), 1000);
        } else {
            clearInterval(timer);
        }
        return () => clearInterval(timer);
    }, [loading]);

    const toggleNumber = (num, type) => {
        if (type === "exclude") {
            if (excluded.includes(num)) {
                setExcluded((prev) => prev.filter((n) => n !== num));
            } else {
                if (excluded.length >= 5) return alert("제외 번호는 최대 5개까지 선택할 수 있습니다.");
                setExcluded((prev) => [...prev, num]);
                setIncluded((prev) => prev.filter((n) => n !== num));
            }
        } else {
            if (included.includes(num)) {
                setIncluded((prev) => prev.filter((n) => n !== num));
            } else {
                if (included.length >= 3) return alert("포함 번호는 최대 3개까지 선택할 수 있습니다.");
                setIncluded((prev) => [...prev, num]);
                setExcluded((prev) => prev.filter((n) => n !== num));
            }
        }
    };

    const getLottoNumbers = async () => {
        setLoading(true);
        try {
            const res = await axios.get("http://localhost:8000/lotto/recommend", {
                params: {
                    count,
                    exclude: excluded.join(","),
                    include: included.join(","),
                    type: recommendType,
                    start_round: startRound,
                    end_round: endRound,
                },
            });
            setSets((prev) => (prev.length + res.data.sets.length > 10 ? res.data.sets : [...prev, ...res.data.sets]));
        } catch (error) {
            alert("API 호출 에러: " + (error.message || ""));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="center-container">
            <div className="lotto-panel">
                <div className="lotto-title-wrapper">
                    <div className="lotto-title">로또번호 생성기</div>
                    <a className="modal-icon-btn" onClick={() => setShowModal(true)}>❗</a>
                </div>

                {showModal && (
                    <div className="modal-backdrop" onClick={() => setShowModal(false)}>
                        <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                            <button className="modal-close-btn" onClick={() => setShowModal(false)}>×</button>
                            <h3 style={{ marginBottom: "10px" }}>🎯 이용 가이드</h3>

                            <p><strong>1.</strong> 세트는 한 번에 <strong>최대 10개</strong>까지만 추천할 수 있습니다.</p>

                            <p><strong>2.</strong> 포함할 번호와 제외할 번호는 아래 기준을 따르는 것이 좋습니다.</p>

                            <ul style={{ margin: "10px 0 15px 0", paddingLeft: "20px", listStyleType: "disc" }}>
                                <li><strong>포함 번호</strong>: 최대 <strong>3개</strong>까지 추천</li>
                                <li><strong>제외 번호</strong>: 최대 <strong>5개</strong>까지 추천</li>
                            </ul>

                            <p style={{ marginBottom: "8px" }}>📌 <strong>가장 이상적인 조합 수:</strong></p>
                            <ul style={{ paddingLeft: "20px", listStyleType: "circle" }}>
                                <li>포함 번호: <strong>1~2개</strong></li>
                                <li>제외 번호: <strong>2~4개</strong></li>
                                <li>포함 + 제외 합계: <strong>6개 이하</strong></li>
                            </ul>

                            <p style={{ marginTop: "12px", color: "#c0392b" }}>
                                ⚠️ 위 기준을 초과하면 추천 정확도가 낮아질 수 있습니다.
                            </p>

                            <p><strong>3.</strong> 딥러닝 기반의 조합 추천의 경우, 다소 시간이 걸릴 수 있습니다.</p>
                            <p><strong>* 확실한 테스트 이후에 배포하오니 걱정하지 않으셔도 됩니다.</strong></p>
                        </div>

                    </div>
                )}

                {loading && (
                    <div className="loading-overlay">
                        <div className="loading-spinner" />
                        <p className="loading-text">
                            {elapsed < 5 ? (
                                <>
                                    번호를 신중히 고르고 있습니다.<br />
                                    조금만 기다려주세요.<br />
                                </>
                            ) : (
                                <>
                                    AI가 열심히 계산 중입니다...!<br />
                                    조금만 기다려주세요.<br />
                                </>
                            )}
                            <br />
                            <strong>{elapsed}초</strong>
                        </p>

                    </div>
                )}

                <div className="round-select-row">
                    <select value={startRound} onChange={(e) => setStartRound(Number(e.target.value))} className="round-select">
                        {Array.from({ length: latestRound }, (_, i) => i + 1).map(num => (
                            <option key={num} value={num}>{num}회</option>
                        ))}
                    </select>
                    에서
                    <select value={endRound} onChange={(e) => setEndRound(Number(e.target.value))} className="round-select">
                        {Array.from({ length: latestRound }, (_, i) => i + 1).map(num => (
                            <option key={num} value={num}>{num}회</option>
                        ))}
                    </select>
                    까지
                </div>

                <div className="type-selector">
                    <select value={recommendType} onChange={(e) => setRecommendType(e.target.value)} className="type-select">
                        <option value="normal">기본</option>
                        <option value="stat">통계 기반</option>
                        <option value="dl">딥러닝</option>
                        <option value="ml">머신러닝</option>
                        <option value="greedy">탐욕 기반</option>
                        <option value="opt">조합 최적화</option>
                        <option value="hybrid">종합 전략</option>
                    </select>
                    <label>분석으로</label>
                </div>

                <div className="count-input-row">
                    <input type="number" min="1" max="10" value={count} onChange={(e) => setCount(Number(e.target.value))} className="count-input" />
                    <label>개 조합</label>
                    <button className="recommend-btn" onClick={getLottoNumbers} disabled={loading || count > 10}>
                        {loading ? "로딩 중..." : "추천받기"}
                    </button>
                </div>

                {sets.length > 0 && (
                    <div>
                        <h2 className="recommend-title">추천 결과</h2>
                        {sets.map((nums, i) => (
                            <div key={i} className="result-set">
                                {nums.slice().sort((a, b) => a - b).map(num => (
                                    <span key={num} className={`lotto-ball ${getBallColorClass(num)}`}>{num}</span>
                                ))}
                            </div>
                        ))}
                    </div>
                )}

                <div className="title-included">포함 번호</div>
                <table className="lotto-table">
                    <tbody>
                    {numbersGrid.map((row, idx) => (
                        <tr key={idx}>
                            {row.filter(n => n <= 45).map(num => (
                                <td key={num}>
                                    <div
                                        className={`number-wrap ${included.includes(num) ? "on" : ""}`}
                                        onClick={() => toggleNumber(num, "include")}
                                    >
                                        <span className="number">{num}</span>
                                    </div>
                                </td>
                            ))}
                            {idx === 6 && (
                                <>
                                <td></td>
                                <td></td>
                                <td></td>
                                <td>
                                    <div
                                        className="number-wrap reset-cell"
                                        onClick={() => setIncluded([])}
                                    >
                                        <span className="number">취소</span>
                                    </div>
                                </td>
                                </>
                            )}

                        </tr>
                    ))}
                    </tbody>
                </table>

                <div className="title-excluded">제외 번호</div>
                <table className="lotto-table">
                    <tbody>
                    {numbersGrid.map((row, idx) => (
                        <tr key={idx}>
                            {row.filter(n => n <= 45).map(num => (
                                <td key={num}>
                                    <div
                                        className={`number-wrap ${excluded.includes(num) ? "on" : ""}`}
                                        onClick={() => toggleNumber(num, "exclude")}
                                    >
                                        <span className="number">{num}</span>
                                    </div>
                                </td>
                            ))}
                            {idx === 6 && (
                                <>
                                <td></td>
                                <td></td>
                                <td></td>
                                <td>
                                    <div
                                        className="number-wrap reset-cell"
                                        onClick={() => setExcluded([])}
                                    >
                                        <span className="number">취소</span>
                                    </div>
                                </td>
                                </>
                            )}
                        </tr>
                    ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

export default App;
