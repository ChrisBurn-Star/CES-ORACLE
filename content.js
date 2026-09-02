/*
 * Edit this file to replace all content on the homepage.
 * Keep the quotation marks and commas. Use full URLs beginning with https://.
 */
window.PROFILE_DATA = {
  name: "Jixian Zhou",
  shortName: "JIXIAN ZHOU",
  initials: "JZ",
  role: "Advertising LLM Algorithm Engineer · Researcher",
  status: "OPEN TO RESEARCH COLLABORATIONS",
  location: "China · Beijing",
  timezone: "Asia/Shanghai",
  email: "1183519798@qq.com",

  intro:
    "I study the architectures, training methods, and theoretical foundations of large language models and foundation models for recommendation.",

  basicInfo: [
    { label: "Current role", value: "Advertising LLM Algorithm Engineer" },
    { label: "Affiliation", value: "Kuaishou Technology" },
    { label: "Focus", value: "LLMs · RECSYS · REPRESENTATION" },
  ],

  aboutLead: "I am an Advertising LLM Algorithm Engineer at Kuaishou, working at the intersection of large language models, recommender systems, and representation learning.",
  aboutParagraphs: [
    "My research centers on efficient, principled Mixture-of-Experts systems for large language models. In Oracle-MoE (ICML 2025), I developed a locality-preserving routing method that reduces expert swapping and accelerates inference under tight memory constraints. In Once Read is Enough (NeurIPS 2024), I studied cluster-guided sparse experts that enable language models to capture long-tail domain knowledge without additional domain-specific pretraining.",
    "At Kuaishou, I connect fundamental questions in representation learning and MoE dynamics with large-scale advertising and recommendation systems. I am particularly interested in architectures and training strategies that make foundation models more scalable, efficient, and effective in real-world applications.",
  ],
  motto: "Stay curious, keep building, and make complex ideas clear.",

  researchKeywords: ["MIXTURE-OF-EXPERTS", "LLM ARCHITECTURE", "TRAINING OPTIMIZATION", "RECOMMENDATION MODELS", "REPRESENTATION LEARNING", "FOUNDATIONS"],
  research: [
    {
      code: "R.01",
      title: "Mixture-of-Experts (MoE)",
      english: "FOUNDATIONS & ARCHITECTURE",
      description: "Studying the theoretical foundations of Mixture-of-Experts models and designing architectures for scalable training, efficient routing, and memory-efficient inference.",
      tags: ["MoE Theory", "Architecture", "Efficient Routing"],
    },
    {
      code: "R.02",
      title: "LLM Training Optimization",
      english: "TRAINING & ALIGNMENT",
      description: "Improving the stability and efficiency of pre-training and post-training through better learning objectives, data strategies, and optimization methods.",
      tags: ["Pre-training", "Post-training", "Optimization"],
    },
    {
      code: "R.03",
      title: "Foundation Models for Recommendation",
      english: "ARCHITECTURE & TRAINING",
      description: "Developing foundation-model architectures and training strategies for recommendation, with an emphasis on scalable user modeling and transferable representations.",
      tags: ["Recommender Systems", "Foundation Models", "User Modeling"],
    },
    {
      code: "R.04",
      title: "Representation Learning",
      english: "LEARNING REPRESENTATIONS",
      description: "Learning robust, generalizable representations from heterogeneous data to support retrieval, recommendation, reasoning, and cross-domain transfer.",
      tags: ["Embeddings", "Multimodal Learning", "Transfer"],
    },
    {
      code: "R.05",
      title: "Foundations of Large Language Models",
      english: "THEORY & UNDERSTANDING",
      description: "Investigating the principles behind language-model capabilities, including scaling behavior, generalization, optimization dynamics, and emergent phenomena.",
      tags: ["Scaling Laws", "Generalization", "Learning Theory"],
    },
  ],

  articles: [
    {
      number: "01",
      title: "Oracle-MoE: Locality-preserving Routing in the Oracle Space for Memory-constrained Large Language Model Inference",
      summary: "Oracle-MoE routes tokens in a compact oracle space to preserve semantic locality across consecutive tokens, reducing expert swapping and accelerating MoE inference on memory-constrained devices without compromising task performance.",
      meta: "2025 · ICML",
      category: "CONFERENCE PAPER",
      url: "https://icml.cc/virtual/2025/poster/43606",
    },
    {
      number: "02",
      title: "Once Read is Enough: Domain-specific Pretraining-free Language Models with Cluster-guided Sparse Experts for Long-tail Domain Knowledge",
      summary: "This work introduces Cluster-guided Sparse Experts, a lightweight architecture that actively learns long-tail domain knowledge during general pretraining and enables strong domain performance without additional domain-specific pretraining.",
      meta: "2024 · NEURIPS",
      category: "CONFERENCE PAPER",
      url: "https://proceedings.neurips.cc/paper_files/paper/2024/hash/a1f12d8d3cc1b4789ff4ebec46e27609-Abstract-Conference.html",
    },
  ],

  education: [
    {
      period: "MASTER'S",
      school: "Fudan University",
      degree: "M.S. in Computer Software and Theory",
      description: "Graduate studies in computer software, learning systems, and the theoretical foundations of machine intelligence.",
    },
  ],

  contactNote: "If you are interested in my research, writing, or a potential collaboration, feel free to reach out.",
};
