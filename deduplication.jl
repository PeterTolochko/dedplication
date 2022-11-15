using PyCall
using DataFrames
using CSV
using SparseArrays

# use this if you can't load spacy into julia
# ENV["PYTHON"] = "/Users/petrotolochko/anaconda3/bin/python"

my_data = DataFrame(CSV.File("data.csv"))

const spacy = pyimport("spacy")
nlp = spacy.load("de_core_news_sm")


function squared_sum(x)
    round(√(sum([a * a for a ∈ x])), digits = 3)
end


function cosine_similarity(x, y)
    """return cosine similarity between two lists"""
    numerator = sum(a * b for (a, b) ∈ zip(x, y))
    denominator = squared_sum(x) * squared_sum(y)
    θ = numerator / denominator
    return round(θ, digits = 3)
end


function get_word_vectors(text)
    processed_texts = nlp.(text)
    word_vectors = [x.vector for x ∈ processed_texts]
    return word_vectors
end


function get_sim_matrix(word_vectors)
    sim_matrix = zeros(length(word_vectors), length(word_vectors))
    for (i, vector_i) ∈ enumerate(word_vectors)
        for (j, vector_j) ∈ enumerate(word_vectors)
            if i < j
                sim_matrix[i, j] = cosine_similarity(vector_i, vector_j)
            end
        end
    end
    return sim_matrix
end


function get_duplicate_ids(text, θ = .995)

    word_vectors = get_word_vectors(text)
    sim_matrix = get_sim_matrix(word_vectors)

    sim_ids = []

    for i ∈ 1:size(sim_matrix)[1], j ∈ 1:size(sim_matrix)[1]
        if i < j
            if sim_matrix[i, j] >= θ
                push!(sim_ids, [i, j])
            end
        end
    end

    sim_ids = [sim_ids[x][2] for x in range(1, length(sim_ids))]
    sort!(sim_ids)
    unique!(sim_ids)

    return sim_ids
end

sim_ids = get_duplicate_ids(my_data[!, "text"])

# @btime get_duplicate_ids(my_data[!, "text"])


filtered_apa = my_data[Not(sim_ids), :] 

CSV.write("", filtered_apa)

#   86.869 s (121631297 allocations: 12.57 GiB)
