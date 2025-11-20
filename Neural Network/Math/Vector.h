#pragma once
#include <array>


namespace math {
	template <std::size_t N>
	struct Vector {

		std::array<float, N> m_value;
		float& operator[](std::size_t i) { return m_value[i]; }
		const float& operator[](std::size_t i) const { return m_value[i]; }
	};
}
